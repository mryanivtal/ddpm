import torch.nn as nn
from .conv_bn_lrelu_block import ConvBnLreluBlock


class DcganDiscriminator(nn.Module):
    '''
    DC-GAN generator module
    '''
    def __init__(self, num_channels, num_feature_maps)-> None:
        super(DcganDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(num_channels, num_feature_maps, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            ConvBnLreluBlock(num_feature_maps, num_feature_maps * 2, 4, stride=2, padding=1, bias=False),
            ConvBnLreluBlock(num_feature_maps * 2, num_feature_maps * 4, 4, stride=2, padding=1, bias=False),
            ConvBnLreluBlock(num_feature_maps * 4, num_feature_maps * 8, 4, stride=2, padding=1, bias=False),
            nn.Conv2d(num_feature_maps * 8, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.discriminator(input)

