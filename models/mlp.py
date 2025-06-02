'''
classification-only implementation of the counting network that 
uses shared MLP for all  
'''

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
from .p2pnet import SetCriterion_Crowd
from .classification import *


class Linear(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Linear, self).__init__()


        self.lin = nn.Sequential(
            nn.Linear(in_feat, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_feat),
        )

    def forward(self, x):

        x = x.permute([0,2,3,1])
        x = x.flatten(start_dim=1, end_dim=2)

        out = self.lin(x)
        
        return out


class MLP_Classifier(nn.Module):
    """MLP classifier-only predictor on top of FPN feature space"""

    def __init__(self, backbone, num_classes, row=2, line=2):
        super().__init__()

        self.vgg_backbone = backbone
        self.num_classes = num_classes + 1
        self.row = row
        self.line = line

        num_anchor_points = row * line

        self.linear = Linear(
            in_feat=256,
            out_feat=self.num_classes,
        )

        self.anchor_points = AnchorPoints(
            pyramid_levels=[
                3,
            ],
            row=row,
            line=line,
        )

        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples: NestedTensor):
        # get the backbone (vgg) features
        features = self.vgg_backbone(samples)
        # construct the feature space
        features_fpn = self.fpn([features[1], features[2], features[3]])
        batch_size = features[0].shape[0]

        # pass through the classifer
        classification = self.linear(features_fpn[1])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        
        output_coord = anchor_points
        output_class = classification
        out = {"pred_logits": output_class, "pred_points": output_coord}

        return out

class MLP(nn.Module):
    "MLP model for both regression and classification tasks"    
  
    def __init__(self, backbone, num_classes, row=2, line=2):
        super().__init__()

        self.vgg_backbone = backbone
        self.num_classes = num_classes + 1
        self.row = row
        self.line = line

        num_anchor_points = row * line

        self.lin_class = Linear(
            in_feat=256,
            out_feat=self.num_classes,
        )

        self.lin_reg = Linear(
            in_feat=256,
            out_feat=self.num_classes,
        )

        self.anchor_points = AnchorPoints(
            pyramid_levels=[
                3,
            ],
            row=row,
            line=line,
        )

        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples: NestedTensor):
        # get the backbone (vgg) features
        features = self.vgg_backbone(samples)
        # construct the feature space
        features_fpn = self.fpn([features[1], features[2], features[3]])
        batch_size = features[0].shape[0]

        # pass sample through classification and regression branch
        classification = self.lin_class(features_fpn[1])
        regression = self.lin_reg(features_fpn[1])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        
        output_coord = regression + anchor_points
        output_class = classification

        out = {"pred_logits": output_class, "pred_points": output_coord}

        return out


def build_mlp(args, training):

    backbone = build_backbone(args)

    # model selection logic
    if args.mlp_classifier:
        model = MLP_Classifier(backbone, args.num_classes, args.row, args.line)
    elif args.mlp:
        model = MLP(backbone, args.num_classes, args.row, args.line)
    
    if not training:
        return model

    weight_dict = {
        "loss_ce": args.label_loss_coef,
        "loss_point": args.point_loss_coef,
        "loss_dense": args.dense_loss_coef,
        "loss_distance": args.distance_loss_coef,
        "loss_count": args.count_loss_coef,
    }

    losses = args.loss
    matcher = build_matcher_crowd(args)
    criterion = SetCriterion_Crowd(
        args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=args.eos_coef,
        ce_coef=args.ce_coef,
        map_res=args.map_res,
        gauss_kernel_res=args.gauss_kernel_res,
        losses=losses,
    )

    return model, criterion 
