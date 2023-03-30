import torch
import torch.nn as nn
from .unet_parts import SinusoidalPositionEmbedding, Block


class SimpleUnet(nn.Module):
    '''
    Unet module
    '''
    def __init__(self, image_channels, out_dim=1, time_emb_dim=32) -> None:
        super(SimpleUnet, self).__init__()

        down_channels = [64, 128, 256, 512, 1024]
        up_channels = list(reversed(down_channels))

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], kernel_size=3, padding=1)
        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)])
        self.output = nn.Conv2d(up_channels[-1], image_channels, out_dim)

    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)
        # initial conv
        x = self.conv0(x)

        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)

        for up in self.ups:
            # Concatenate residual inputs as additional channels
            residual_x = residual_inputs.pop()
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)




