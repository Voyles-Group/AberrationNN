from torch import nn
import torch.nn.functional as F
import torch
from typing import List
from AberrationNN.covpadsame import Conv2dSame, ConvTranspose2dSame


class Standardization_Layer(nn.Module):
    """Standardization layer. Normalizes the input tensor to have zero mean and unit variance"""

    def __init__(self):
        super(Standardization_Layer, self).__init__()

    # def flatten(self, x: torch.Tensor) -> torch.Tensor:
    #     # this is used for enhancing data in the dark field, which is not useful here. 
    #     return x**0.1

    def standardize(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x, dim=[1, 2, 3], keepdims=True)
        x_var = torch.var(x, dim=[1, 2, 3], keepdims=True)  # standardized for each image in the batch seperately
        # x_std = torch.maximum(torch.sqrt(x_var), torch.full(x_var.shape, 1e-8))# this cause the tensor not on same
        # device if using cuda.
        x_std = torch.clamp(torch.sqrt(x_var), min=1e-8)
        x = torch.divide(torch.subtract(x, x_mean), x_std)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.standardize(x)


class Convolution_Block(nn.Module):
    """
    conv block implementing 2D-Convolution, normalization and activation.
    Tunable parameter: dropout, activation(0,1,2)
    """

    def __init__(self, input_channels=None, output_channels=None, nb_layers=1, kernel_size=3, stride=1, padding=0,
                 activation=0, dropout=0.05, transpose=False):
        super(Convolution_Block, self).__init__()
        self.activation = activation
        self.transpose = transpose

        if self.activation == 0:
            self.activation_layer = nn.LeakyReLU()
        elif self.activation == 1:
            self.activation_layer = nn.ReLU()
        elif self.activation == 2:
            self.activation_layer = nn.SiLU()  # swish activation
        else:
            raise Exception("pick activation 0 1 2 for leakyrelu, relu and swish")

        block = []
        for idx in range(nb_layers):
            input_channels = output_channels if idx > 0 else input_channels
            if self.transpose:
                block.append(
                    ConvTranspose2dSame(input_channels, output_channels, kernel_size=kernel_size, stride=stride,
                                        padding=padding, dtype=torch.float32))
            else:
                block.append(
                    Conv2dSame(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                               dtype=torch.float32))

            if dropout > 0:
                block.append(nn.Dropout(dropout))

            block.append(self.activation_layer)

            block.append(nn.BatchNorm2d(output_channels))

        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.block(x)
        return output


class Conv_Stack(nn.Module):
    """
    Layer implementing a sequence of Convolution Blocks(2D-Convolution, normalization and activation).
    Tunable parameter: n_blocks

    """

    def __init__(self, n_blocks=2, input_channels=None, output_channels=None, nb_layers=1, kernel_size=3, stride=1,
                 padding=0, activation=0, dropout=0.05):
        super(Conv_Stack, self).__init__()

        self.n_blocks = n_blocks
        self.conv_blocks = nn.ModuleList()

        for b in range(self.n_blocks):
            if b == 0:
                self.conv_blocks.append(
                    Convolution_Block(input_channels=input_channels, output_channels=output_channels, padding=padding,
                                      activation=activation, dropout=dropout))
            else:
                self.conv_blocks.append(
                    Convolution_Block(input_channels=output_channels, output_channels=output_channels, padding=padding,
                                      activation=activation, dropout=dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.conv_blocks:
            x = block(x)
        return x


class Contraction_Path(nn.Module):
    """
    Downsampling path combining the Convolution Stack and a `n` strided convolution layer per floor, 
    raising the number of filters by a factor of `n`.
    Output: bottleneck output and the convolution layers for skip connections, listed from shallow to deep. 
    """

    def __init__(self, init_channel = 2, depth=3, n_blocks=2, first_inputchannels=32, activation=0, dropout=0.05):
        super(Contraction_Path, self).__init__()
        self.init_channel = init_channel
        self.depth = depth
        self.activation = activation
        self.dropout = dropout
        self.block = nn.ModuleList()  # the normal list will not be registered correctly as submodules of the model
        # and have empty parameters.
        channels = [init_channel, first_inputchannels, first_inputchannels * 2, first_inputchannels * 4, first_inputchannels * 8]
        for l in range(self.depth):
            if l == 0:
                # floor 0 at the top
                self.block.append(Conv_Stack(n_blocks=n_blocks, input_channels=channels[0], output_channels=channels[1],
                                             activation=activation, dropout=dropout))
                # down to floor 1 double nfilters and also downsampling
                self.block.append(Convolution_Block(input_channels=channels[1], output_channels=channels[2], stride=2,
                                                    activation=activation, dropout=dropout))
            elif l == (self.depth - 1):
                # floor final, no more downsampling
                self.block.append(
                    Conv_Stack(n_blocks=n_blocks, input_channels=channels[l + 1], output_channels=channels[l + 1],
                               activation=activation, dropout=dropout))
            else:
                self.block.append(
                    Conv_Stack(n_blocks=n_blocks, input_channels=channels[l + 1], output_channels=channels[l + 1],
                               activation=activation, dropout=dropout))
                self.block.append(
                    Convolution_Block(input_channels=channels[l + 1], output_channels=channels[l + 2], stride=2,
                                      activation=activation, dropout=dropout))

        # self.blockmodule = nn.Sequential(*self.block)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        output = []
        useful_out = []
        for bk in self.block:
            x = bk(x)
            output.append(x)
        for i, out in enumerate(output):
            if not (i % 2):
                useful_out.append(out)
        return useful_out


class Expansion_Path(nn.Module):
    """
    Upsampling path combining the Convolution Stacks, concatenation at each floor and convtranspose layers 
    for reducing the number of filters and upsampling, 
    """

    def __init__(self, depth=3, n_blocks=2, first_inputchannels=32, activation=0, dropout=0.05):
        super(Expansion_Path, self).__init__()
        self.depth = depth
        self.activation = activation
        self.dropout = dropout

        # lets say the bottom is floor 0
        # cbt: convolution block transpose cs: convolution stack
        self.cbt_0_1 = Convolution_Block(input_channels=first_inputchannels * (2 ** (depth - 1)),
                                         output_channels=first_inputchannels * (2 ** (depth - 2)), stride=2,
                                         activation=activation, dropout=dropout, transpose=True)
        self.cs_1 = Conv_Stack(n_blocks=n_blocks, input_channels=first_inputchannels * (2 ** (depth - 1)),
                               output_channels=first_inputchannels * (2 ** (depth - 2)), activation=activation,
                               dropout=dropout)

        self.cbt_1_2 = Convolution_Block(input_channels=first_inputchannels * (2 ** (depth - 2)),
                                         output_channels=first_inputchannels * (2 ** (depth - 3)), stride=2,
                                         activation=activation, dropout=dropout, transpose=True)
        # index from left to right
        self.cs_2_2 = Conv_Stack(n_blocks=n_blocks, input_channels=first_inputchannels * (2 ** (depth - 3)) * 3,
                                 output_channels=first_inputchannels * (2 ** (depth - 3)), activation=activation,
                                 dropout=dropout)
        self.cs_2_1 = Conv_Stack(n_blocks=n_blocks, input_channels=first_inputchannels * (2 ** (depth - 3)) * 2,
                                 output_channels=first_inputchannels * (2 ** (depth - 3)), activation=activation,
                                 dropout=dropout)
        if self.depth == 4:
            self.cbt_2_3 = Convolution_Block(input_channels=first_inputchannels * (2 ** (depth - 3)),
                                             output_channels=first_inputchannels * (2 ** (depth - 4)), stride=2,
                                             activation=activation, dropout=dropout, transpose=True)
            self.cs_3_3 = Conv_Stack(n_blocks=n_blocks, input_channels=first_inputchannels * (2 ** (depth - 4)) * 4,
                                     output_channels=first_inputchannels * (2 ** (depth - 4)), activation=activation,
                                     dropout=dropout)
            self.cs_3_2 = Conv_Stack(n_blocks=n_blocks, input_channels=first_inputchannels * (2 ** (depth - 4)) * 3,
                                     output_channels=first_inputchannels * (2 ** (depth - 4)), activation=activation,
                                     dropout=dropout)
            self.cs_3_1 = Conv_Stack(n_blocks=n_blocks, input_channels=first_inputchannels * (2 ** (depth - 4)) * 2,
                                     output_channels=first_inputchannels * (2 ** (depth - 4)), activation=activation,
                                     dropout=dropout)

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        k = self.cbt_0_1(x[-1])
        f1 = torch.cat([x[-2], self.cbt_0_1(x[-1])], 1)
        f1_c = self.cs_1(f1)

        f2_1 = torch.cat([x[-3], self.cbt_1_2(x[-2])], 1)
        f2_1_c = self.cs_2_1(f2_1)
        f2_2 = torch.cat([f2_1_c, x[-3], self.cbt_1_2(f1_c)], 1)
        f2_2_c = self.cs_2_2(f2_2)
        if self.depth == 4:
            f3_1 = torch.cat([x[-4], self.cbt_2_3(x[-3])], 1)
            f3_1_c = self.cs_3_1(f3_1)
            f3_2 = torch.cat([f3_1_c, x[-4], self.cbt_2_3(f2_1_c)], 1)
            f3_2_c = self.cs_3_2(f3_2)
            f3_3 = torch.cat([f3_2_c, f3_1_c, x[-4], self.cbt_2_3(f2_2_c)], 1)
            f3_3_c = self.cs_3_3(f3_3)

            return f3_3_c
        else:
            return f2_2_c
