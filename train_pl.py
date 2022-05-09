import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.loader import ImageDataModule
from models import LightningVQVAE
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.strategies.ddp import DDPStrategy

def main(hparams):
    config_dir = 'models/configs/VQVAE.yaml'
    with open(config_dir) as fin:
        config = yaml.safe_load(fin)["VQVAE"]

    print(config)
    model = LightningVQVAE(config, hparams)
    print("hparams: ", hparams)
    data_module = ImageDataModule.from_argparse_args(hparams)
    # Add progress bar
    callbacks = [TQDMProgressBar(refresh_rate=10)]
    # got warning about checking for unused parameters and it potentially slowing down training
    # hence it was turned off
    trainer = Trainer.from_argparse_args(hparams, log_every_n_steps=2, callbacks=callbacks, plugins=DDPStrategy(find_unused_parameters=False))
    trainer.fit(model, train_dataloaders=data_module)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4)
    parser = ImageDataModule.add_argparse_args(parser)
    print("parser: ", parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
