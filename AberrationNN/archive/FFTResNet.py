from torch import nn
import torch.nn.functional as F
import torch


class ResBlock(nn.Module):
    """
    Builds a residual block

    Args:
        ndim:
            Data dimensionality (1D or 2D)
        input_channels:
            Number of input channels for the block
        output_channels:
            Number of the output channels for the block
        batch_norm:
            Add batch normalization to each layer in the block
        activation:
        dropout:

    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 batch_norm: bool = True,
                 activation: int = 0,
                 dropout: float = 0,
                 ) -> None:

        super(ResBlock, self).__init__()
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout

        self.c0 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.c1 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1)
        if self.batch_norm:
            bn = nn.BatchNorm2d
            self.bn1 = bn(output_channels)
            self.bn2 = bn(output_channels)

        if self.activation == 0:
            self.activation_layer = nn.LeakyReLU()
        elif self.activation == 1:
            self.activation_layer = nn.ReLU()
        elif self.activation == 2:
            self.activation_layer = nn.SiLU()  # swish activation
        else:
            raise Exception("pick activation 0 1 2 for leakyrelu, relu and swish")

        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines forward pass
        """
        x = self.c0(x)
        residual = x
        out = self.c1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.activation_layer(out)
        out = self.c2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out += residual
        out = self.activation_layer(out)
        if self.dropout > 0:
            out = self.dropout_layer(out)

        return out


class FFTResNet(nn.Module):

    def __init__(self, first_inputchannels=32, activation=0, dropout=0.05):
        super(FFTResNet, self).__init__()
        self.dropout = dropout
        self.activation = activation

        self.resblock1 = ResBlock(input_channels=first_inputchannels, output_channels=first_inputchannels * 2,
                                  batch_norm=True, activation=self.activation, dropout=self.dropout)
        self.resblock2 = ResBlock(input_channels=first_inputchannels * 2, output_channels=first_inputchannels * 4,
                                  batch_norm=True, activation=self.activation, dropout=self.dropout)
        self.resblock3 = ResBlock(input_channels=first_inputchannels * 4, output_channels=first_inputchannels * 8,
                                  batch_norm=True, activation=self.activation, dropout=self.dropout)
        self.resblock4 = ResBlock(input_channels=first_inputchannels * 8, output_channels=first_inputchannels * 16,
                                  batch_norm=True, activation=self.activation, dropout=self.dropout)
        self.flatten = nn.Flatten()
        bottleneck_size = 8  # 32x64 down-sampled to 4x2
        self.dense1 = nn.Linear(first_inputchannels * 16 * bottleneck_size, first_inputchannels * 8 * bottleneck_size)
        self.dense2 = nn.Linear(first_inputchannels * 8 * bottleneck_size, first_inputchannels * 2 * bottleneck_size)
        self.dense3 = nn.Linear(1 + first_inputchannels * 2 * bottleneck_size, 9)

    def forward(self, x: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        c1 = self.resblock1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        c2 = self.resblock2(d1)
        d2 = F.max_pool2d(c2, kernel_size=2, stride=2)
        c3 = self.resblock3(d2)
        d3 = F.max_pool2d(c3, kernel_size=2, stride=2)
        c4 = self.resblock4(d3)
        d4 = F.max_pool2d(c4, kernel_size=2, stride=2)
        final = self.flatten(d4)
        final = F.softmax(self.dense1(final))
        final = F.softmax(self.dense2(final)) # you didn't add activation for these intermediate layers , which is wrong
        final = torch.cat([final, cov], dim=1)
        final = self.dense3(final)

        return final


class WFTResNetFull(nn.Module):

    def __init__(self, first_inputchannels=256, activation=0, dropout=0.05, dense_layer=3, first_order_only=False):
        super(WFTResNetFull, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.dense_layer = dense_layer
        self.first_order_only = first_order_only
        self.resblock1 = ResBlock(input_channels=first_inputchannels, output_channels=first_inputchannels * 2,
                                  batch_norm=True, activation=self.activation, dropout=self.dropout)
        self.resblock2 = ResBlock(input_channels=first_inputchannels * 2, output_channels=first_inputchannels * 4,
                                  batch_norm=True, activation=self.activation, dropout=self.dropout)
        self.resblock3 = ResBlock(input_channels=first_inputchannels * 4, output_channels=first_inputchannels * 8,
                                  batch_norm=True, activation=self.activation, dropout=self.dropout)
        self.resblock4 = ResBlock(input_channels=first_inputchannels * 8, output_channels=first_inputchannels * 16,
                                  batch_norm=True, activation=self.activation, dropout=self.dropout)
        self.flatten = nn.Flatten()
        bottleneck_size = 8  # 32x64 down-sampled to 4x2
        if self.dense_layer == 3:
            self.dense1 = nn.Linear(first_inputchannels * 16 * bottleneck_size, first_inputchannels * bottleneck_size)
            self.dense2 = nn.Linear(first_inputchannels * bottleneck_size, bottleneck_size * 16)
            if self.first_order_only:
                self.dense3 = nn.Linear(1 + bottleneck_size * 16, 4)
            else:
                self.dense3 = nn.Linear(1 + bottleneck_size * 16, 9)
        elif self.dense_layer == 4:
            self.dense1 = nn.Linear(first_inputchannels * 16 * bottleneck_size,
                                    first_inputchannels * bottleneck_size * 4)
            self.dense2 = nn.Linear(first_inputchannels * bottleneck_size * 4, first_inputchannels * bottleneck_size)
            self.dense3 = nn.Linear(first_inputchannels * bottleneck_size, bottleneck_size * 16)

            if self.first_order_only:
                self.dense4 = nn.Linear(1 + bottleneck_size * 16, 4)
            else:
                self.dense4 = nn.Linear(1 + bottleneck_size * 16, 9)
        else:
            raise Exception("Available dense_layer 3 or 4")

    def forward(self, x: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        c1 = self.resblock1(x)
        d1 = F.max_pool2d(c1, kernel_size=2, stride=2)
        c2 = self.resblock2(d1)
        d2 = F.max_pool2d(c2, kernel_size=2, stride=2)
        c3 = self.resblock3(d2)
        d3 = F.max_pool2d(c3, kernel_size=2, stride=2)
        c4 = self.resblock4(d3)
        d4 = F.max_pool2d(c4, kernel_size=2, stride=2)
        final = self.flatten(d4)
        final = self.dense1(final)
        final = self.dense2(final)
        if self.dense_layer == 3:
            final = torch.cat([final, cov], dim=1)
            final = self.dense3(final)
        else:
            final = self.dense3(final)
            final = torch.cat([final, cov], dim=1)
            final = self.dense4(final)
        return final
