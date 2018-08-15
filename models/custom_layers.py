# from https://github.com/nashory/pggan-pytorch/blob/master/custom_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import copy
from torch.nn.init import kaiming_normal, calculate_gain
from torch.nn import Sequential as nnSequential


# same function as ConcatTable container in Torch7.
# Note: not the same as layer type common.Concat: ConcatTable returns a tuple of Tensors
# while Concat returns a concatenated Tensor. ConcatTable should be used before a fadein_layer
# while Concat is for skip connections.
class ConcatTable(nn.Module):
    def __init__(self, layer1, layer2):
        super(ConcatTable, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        y = [self.layer1(x), self.layer2(x)]
        return y


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class TruncateChannels(nn.Module):
    def __init__(self, layer, n_channels):
        super().__init__()
        self.layer = layer
        self.n_channels = n_channels  # number of channels to keep

    def forward(self, x):
        return self.layer.forward(x[:, :self.n_channels])


class fadein_layer(nn.Module):
    def __init__(self):
        super(fadein_layer, self).__init__()
        self.alpha = 0.0

    def update_alpha(self, value):
        self.alpha = value
        self.alpha = max(0, min(self.alpha, 1.0))

    # input : [x_low, x_high] from ConcatTable()
    def forward(self, x):
        return torch.add(x[0].mul(1.0-self.alpha), x[1].mul(self.alpha))

    def __repr__(self):
        tmpstr = '{}(alpha={}) (\n'.format(self.__class__.__name__, self.alpha)
        for key, module in self._modules.items():
            modstr = module.__repr__()
            modstr = _addindent(modstr, 2)
            tmpstr = tmpstr + '  (' + key + '): ' + modstr + '\n'
        tmpstr = tmpstr + ')'
        return tmpstr
