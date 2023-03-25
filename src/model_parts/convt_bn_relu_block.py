import torch
import torch.nn as nn


class ConvtBnReluBlock(nn.Module):
    '''
    Conv2d-BatchNorm-Relu block
    Input (named) arguments - same as ConvTranspose2d arguments:
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    '''

    def __init__(self, in_channels, out_channels, kernel_size, **kwargs) -> None:
        super(ConvtBnReluBlock, self).__init__()

        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, input):
        return self.block(input)

