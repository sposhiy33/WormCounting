import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import (NestedTensor, accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized,
                       nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .matcher import build_matcher_crowd

class ClassifierModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points, num_classes=2, feature_size=256):
        super(ClassifierModel, self).__init__()
 
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    def forward(self, x):
       
        # forward pass through specific convolutional layer
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.act2(out)
        out = self.output(out)

class RegressionModel(nn.Module):
    pass

class MCCModel():
    pass
