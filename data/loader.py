    # Originally found in https://github.com/lucidrains/DALLE-pytorch
from pathlib import Path
from random import randint, choice

import pandas as pd
import os
import pickle
import lmdb
import PIL
import argparse
import torch
import json
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from pytorch_lightning import LightningDataModule

class ImageData(Dataset):
    def __init__(self,
                 folder: str,
                 wsi_names: list,
                 shuffle=False,
                 image_size=224
                 ):
        """Create a text image dataset from a directory with WSIs containing patches.

        Args:
            folder (str): Path to folder containing the LMDB WSI
            wsi_names (str): Path to .csv folder containing the complete (cross-validation) data
            image_size (int, optional): The size of outputted images. Defaults to 224.
        """
        super().__init__()
        self.image_size = image_size
        self.wsi_names = wsi_names
        self.shuffle = shuffle

        print("Number of Images: ", len(self.wsi_names))
        self.folder = folder

        # keep the data augmentation simple
        # Niccolo's albumentations pipeline was not going well
        self.image_transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            RotateDegrees(),
            T.ToTensor(),
            T.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))
        ])

        print("All the data ready to go")

    def __len__(self):
        return len(self.wsi_names)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, index):

        try:
            key = self.wsi_names[index]
        except:
            return self.skip_sample(index)

        lmdb_file_root = os.path.join(self.folder, key)
        try:
            env = lmdb.open(str(Path(self.folder) / f"{Path(key)}-lmdb"), readonly=True)
        except:
            return self.skip_sample(index)

        entries = env.stat()["entries"]

        if entries == 0:
            return self.skip_sample(ind)

        # this one is still random but we keep it as such for now
        # TODO: if shuffle false no random chosen patches.
        chosen_patch = np.random.randint(0,entries,1)[0]

        # Start a new read transaction
        with env.begin() as txn:
            # Read all images in one single transaction, with one lock
            data = txn.get(f"{chosen_patch:08}".encode("ascii"))
            image = pickle.loads(data)
            img = np.frombuffer(image, dtype=np.uint8).reshape((self.image_size,self.image_size,3))
        env.close()

        img = self.image_transform(PIL.Image.fromarray(img))

        return img

class RotateDegrees:
    def __init__(self, angles=[-180, -90, 90, 180, 0]):
        self.angles = angles

    def __call__(self, x):
        angle = choice(self.angles)
        return TF.rotate(x, angle, TF.InterpolationMode.BILINEAR)

class ImageDataModule(LightningDataModule):
    def __init__(self,
                 folder: str,
                 data_csv: str,
                 batch_size: int,
                 image_size: int,
                 num_workers: int,
                 shuffle=True
                 ):
        """Create an image datamodule"""
        super().__init__()
        self.folder = folder
        self.data_csv = data_csv
        self.num_workers = num_workers
        self.image_size = image_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        train_csv = os.path.join(self.data_csv, "10_cross_validation_gt.csv")
        val_csv = os.path.join(self.data_csv, "ground_truth_AOEC_umc.csv")

        self.train_names = list(pd.read_csv(train_csv, header=None).to_numpy()[:,0])
        self.val_names = list(pd.read_csv(val_csv, header=None).to_numpy()[:,0])

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--folder', type=str, required=True, help='directory of your training folder')
        parser.add_argument('--batch_size', type=int, default=8, help='size of the batch')
        parser.add_argument('--data_csv', type=str, default="./data/cross_validation_folds", help='path to the folder with the .csv data')
        parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
        parser.add_argument('--image_size', type=int, default=224, help='size of the images')
        parser.add_argument('--shuffle', type=bool, default=True, help='whether to use shuffling during sampling')
        return parser

    def setup(self, stage=None):
        self.train_dataset = ImageData(self.folder, self.train_names, image_size=self.image_size, shuffle=True)
        self.val_dataset = ImageData(self.folder, self.val_names, image_size=self.image_size, shuffle=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True, collate_fn=self.dl_collate_fn)

    def val_dataloader(self):
        # fixed batch size (smaller than train batch size) for validation
        return DataLoader(self.val_dataset, batch_size=32, shuffle=False, num_workers=self.num_workers, drop_last=True, collate_fn=self.dl_collate_fn)

    def dl_collate_fn(self, batch):
        #return torch.stack([row[0] for row in batch])
        return torch.stack([row for row in batch])
