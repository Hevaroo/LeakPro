"""Discriminator class from LetheSec/PLG-MI-Attack repository"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn import utils

class SNResNetProjectionDiscriminator(nn.Module):

    def __init__(self, num_features=64, num_classes=0, activation=F.relu):
        super(SNResNetProjectionDiscriminator, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.activation = activation

        self.block1 = OptimizedBlock(3, num_features)
        self.block2 = Block(num_features, num_features * 2,
                            activation=activation, downsample=True)
        self.block3 = Block(num_features * 2, num_features * 4,
                            activation=activation, downsample=True)
        self.block4 = Block(num_features * 4, num_features * 8,
                            activation=activation, downsample=True)
        self.block5 = Block(num_features * 8, num_features * 16,
                            activation=activation, downsample=True)
        self.l6 = utils.spectral_norm(nn.Linear(num_features * 16, 1))
        if num_classes > 0:
            self.l_y = utils.spectral_norm(
                nn.Embedding(num_classes, num_features * 16))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.l6.weight.data)
        optional_l_y = getattr(self, 'l_y', None)
        if optional_l_y is not None:
            init.xavier_uniform_(optional_l_y.weight.data)

    def forward(self, x, y=None):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.block5(h)
        h = self.activation(h)
        # Global pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l6(h)
        if y is not None:
            output += torch.sum(self.l_y(y) * h, dim=1, keepdim=True)
        return output


class Block(nn.Module):

    def __init__(self, in_ch, out_ch, h_ch=None, ksize=3, pad=1,
                 activation=F.relu, downsample=False):
        super(Block, self).__init__()

        self.activation = activation
        self.downsample = downsample

        self.learnable_sc = (in_ch != out_ch) or downsample
        if h_ch is None:
            h_ch = in_ch
        else:
            h_ch = out_ch

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, h_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(h_ch, out_ch, ksize, 1, pad))
        if self.learnable_sc:
            self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        if self.learnable_sc:
            init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        if self.downsample:
            return F.avg_pool2d(x, 2)
        return x

    def residual(self, x):
        h = self.c1(self.activation(x))
        h = self.c2(self.activation(h))
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h


class OptimizedBlock(nn.Module):

    def __init__(self, in_ch, out_ch, ksize=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation

        self.c1 = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize, 1, pad))
        self.c2 = utils.spectral_norm(nn.Conv2d(out_ch, out_ch, ksize, 1, pad))
        self.c_sc = utils.spectral_norm(nn.Conv2d(in_ch, out_ch, 1, 1, 0))

        self._initialize()

    def _initialize(self):
        init.xavier_uniform_(self.c1.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c2.weight.data, math.sqrt(2))
        init.xavier_uniform_(self.c_sc.weight.data)

    def forward(self, x):
        return self.shortcut(x) + self.residual(x)

    def shortcut(self, x):
        return self.c_sc(F.avg_pool2d(x, 2))

    def residual(self, x):
        h = self.activation(self.c1(x))
        return F.avg_pool2d(self.c2(h), 2)