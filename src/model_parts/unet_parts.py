""" Parts of the U-Net model """
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super(SinusoidalPositionEmbedding, self).__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim / 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * (-1) * embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, up=False):
        super(Block, self).__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        if up:
            self.conv1 = nn.Conv2d(2*in_channels, out_channels, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.bnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, t, ):
        # First conv
        h = self.bnorm(self.relu(self.conv1(x)))
        # Add time embedding
        time_emb = self.relu(self.time_mlp(t))
        # extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel to vector
        h = h + time_emb
        # Second conv
        h = self.bnorm(self.relu(self.conv2(h)))
        # Up / Down sample
        return self.transform(h)