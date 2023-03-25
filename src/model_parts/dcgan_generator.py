import torch.nn as nn
from .convt_bn_relu_block import ConvtBnReluBlock


class DcganGenerator(nn.Module):
    """
    DC-GAN generator module
    """
    def __init__(self, num_channels, num_feature_maps, latent_dim) -> None:
        super(DcganGenerator, self).__init__()

        self.num_channels = num_channels
        self.num_feature_maps = num_feature_maps
        self.latent_dim = latent_dim

        self.generator = nn.Sequential(
            ConvtBnReluBlock(latent_dim, num_feature_maps * 8, 4, stride=1, padding=0, bias=False),
            ConvtBnReluBlock(num_feature_maps * 8, num_feature_maps * 4, 4, stride=2, padding=1, bias=False),
            ConvtBnReluBlock(num_feature_maps * 4, num_feature_maps * 2, 4, stride=2, padding=1, bias=False),
            ConvtBnReluBlock(num_feature_maps * 2, num_feature_maps, 4, stride=2, padding=1, bias=False),
            nn.ConvTranspose2d(num_feature_maps, num_channels, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.generator(input)
