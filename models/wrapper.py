import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
import numpy as np
import copy
from .model import VQVAE

class LightningVQVAE(pl.LightningModule):
    def __init__(self,
                 config: dict,
                 hparams
                 ):
        """A lightning wrapper for a VQVAE model.

        Args:
            config (dict): A dictionary containing the VQVAE instantiation parameters.
            hparams (ArgumentParser): A set of hyper parameters
        """
        super().__init__()
        self.model = VQVAE(**config)
        self.beta = config["beta"]
        self.lr = hparams.lr

    def forward(self, imgs):
        return self.model(imgs)

    def training_step(self, train_batch, idx):
        # get optimizers and scheduler
        return self._step(train_batch, idx, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx):
        # TODO: img latents
        x_tilde, z_e_x, z_q_x = self(batch)
        return x_tilde

    def _step(self, batch, batch_idx, stage: str):
        x_tilde, z_e_x, z_q_x = self(batch)
        recon_loss = F.mse_loss(x_tilde, batch)
        vq_loss = F.mse_loss(z_q_x, z_e_x.detach())
        commitment_loss = F.mse_loss(z_e_x, z_q_x.detach())

        self.log(f"{stage}_recon_loss", recon_loss.item(), on_step=True)
        recon_imgs = x_tilde[:8]
        grid = torchvision.utils.make_grid(recon_imgs.cpu(), nrows=2, range=(-1,1), normalize=True)
        self.logger.experiment.add_image("generated_images", grid, 0)

        loss = recon_loss + vq_loss + self.beta * commitment_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=150
        )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
