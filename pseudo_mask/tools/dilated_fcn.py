import functools
import torch.nn as nn
import torch.nn.functional as F
import torch

from utils import get_upsampling_weight, l2_norm
from loss import cross_entropy2d
import math

from skimage.morphology import thin
from scipy import ndimage
import copy
import numpy as np
from utils import freeze_weights, masked_embeddings, weighted_masked_embeddings, compute_weight

# FCN 8s
class dilated_fcn8s(fcn8s):
    def __init__(self, *args, **kwargs):
        super(dilated_fcn8s, self).__init__(*args, **kwargs)

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
#            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, dilation=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, dilation=4, padding=4),
            nn.ReLU(inplace=True),
#            nn.MaxPool2d(2, stride=2, ceil_mode=True),
        )
