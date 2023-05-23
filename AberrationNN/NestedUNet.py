from torch import nn
import torch.nn.functional as F
import torch
from typing import List
from AberrationNN.blocks import Standardization_Layer, Contraction_Path, Expansion_Path


class NestedUNet(nn.Module):

    def __init__(self, depth=3, n_blocks=2, first_inputchannels=32, activation=0, dropout=0.05):
        super(NestedUNet, self).__init__()
        self.depth = depth
        self.activation = activation
        self.dropout = dropout

        self.standardize = Standardization_Layer()
        self.contraction = Contraction_Path(depth=depth, n_blocks=n_blocks, first_inputchannels=first_inputchannels,
                                            activation=activation, dropout=dropout)
        self.expansion = Expansion_Path(depth=depth, n_blocks=n_blocks, first_inputchannels=first_inputchannels,
                                        activation=activation, dropout=dropout)
        self.final = nn.Conv2d(first_inputchannels, 1, 3, 1, padding='same')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] % 4 or x.shape[2] % 4:
            raise Exception('Image width and height need to be a multiple of 4')
        x = self.standardize(x)
        x = self.contraction(x)
        x = self.expansion(x)
        # final layer
        x = self.final(x)
        return x
