from AberrationNN.FCAResNet import *
from torch import nn
import torch


class CombinedNN(nn.Module):
    def __init__(self,
                 first_inputchannels=64, reduction=16,
                 skip_connection=False, fca_block_n=2, if_FT=True, if_CAB=True):
        super(CombinedNN, self).__init__()
        self.first_inputchannels = first_inputchannels
        self.reduction = reduction
        self.skip_connection = skip_connection
        self.fca_block_n = fca_block_n
        self.if_FT = if_FT
        self.if_CAB = if_CAB

        self.firstordermodel = FCAResNet(first_inputchannels=self.first_inputchannels, reduction=self.reduction,
                                         skip_connection=self.skip_connection, fca_block_n=self.fca_block_n,
                                         if_FT=self.if_FT, if_CAB=self.if_CAB)
        self.secondordermodel = FCAResNetSecondOrder(first_inputchannels=self.first_inputchannels,
                                                     reduction=self.reduction,
                                                     skip_connection=self.skip_connection, fca_block_n=self.fca_block_n,
                                                     if_FT=self.if_FT, if_CAB=self.if_CAB)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.firstordermodel(x)
        second = self.secondordermodel(x, first)

        return torch.cat([first, second], dim=1)


