import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from models.functions import vq, vq_st

# CREDIT: https://github.com/ritheshkumar95/pytorch-vqvae


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

class one_by_one_block(nn.Module):
    def __init__(self, dim, target_dim=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, target_dim, 1, 1, 1),
            nn.BatchNorm2d(target_dim),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)

class VQVAE(nn.Module):
    def __init__(self, input_dim, embed_dim, codebook_size=512, codebook_dim=64, beta=0.25):
        super().__init__()
        K = codebook_size
        dim = embed_dim

        # Added an extra layer to compress the image more
        # 224x224 -> 28x28, now
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim//2, 4, 2, 1),
            nn.BatchNorm2d(dim//2),
            nn.LeakyReLU(True),
            nn.Conv2d(dim//2, dim, 4, 2, 1),
            one_by_one_block(dim, target_dim=codebook_dim),
            ResBlock(codebook_dim),
            ResBlock(codebook_dim),
        )

        self.codebook = VQEmbedding(K, codebook_dim)

        self.decoder = nn.Sequential(
            ResBlock(codebook_dim),
            ResBlock(codebook_dim),
            one_by_one_block(codebook_dim, target_dim=dim),
            nn.ConvTranspose2d(dim, dim//2, 4, 2, 1),
            nn.BatchNorm2d(dim//2),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(dim//2, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
