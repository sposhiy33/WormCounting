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

from src.util.misc import (NestedTensor, accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized,
                       nested_tensor_from_tensor_list)

from src.models.backbone import build_backbone
from src.models.matcher import build_matcher_crowd
from src.models.p2pnet import SetCriterion_Crowd
from src.models.classification import *


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
            out_feat=2,
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
    if args.architecture == "mlp_classifier":
        model = MLP_Classifier(backbone, args.num_classes, args.row, args.line)
    elif args.architecture == "mlp":
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
        debris_class_idx=args.debris_class_idx,
        debris_radius=args.debris_radius,
        neg_lambda_debris=args.neg_lambda_debris,
        neg_lambda_other=args.neg_lambda_other,
        normalized_ce=args.normalized_ce,
    )

    return model, criterion


##  ---- Two stage MLP configuration ----

class MLP_TwoStage(nn.Module):

    def __init__(self, stage_one_model_path, backbone, num_classes, row=2, line=2):
        
        
        super().__init__()

        # --- load the stage one model (already pretrained)
        self.stage_one_model = MLP(backbone, num_classes, row, line) # stage one model only has two classes: positive and negative
        checkpoint = torch.load(stage_one_model_path, map_location="cpu")
        self.stage_one_model.load_state_dict(checkpoint["model"])
        self.stage_one_model.eval()
        for param in self.stage_one_model.parameters():
            param.requires_grad = False

        # --- build the stage two model (fine grained classification on positive points from stage one)
        #   
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
            out_feat=2,
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
    

        # --- get positive points from stage one model
        with torch.no_grad():
            stage_one_output = self.stage_one_model(samples)
            positive_mask = F.softmax(stage_one_output["pred_logits"], dim=-1)[:, :, 1] > 0.5

        # --- get features from the backbone
        features = self.vgg_backbone(samples)
        features_fpn = self.fpn([features[1], features[2], features[3]])

        # --- make anchor points
        batch_size = features[0].shape[0]
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)

        # --- get outputs from the second stage model
        classification = self.lin_class(features_fpn[1])
        regression = self.lin_reg(features_fpn[1])

        output_coord = regression + anchor_points
        output_class = classification

        # --- apply positive mask to both outputs ... ensure that only positive points have a chance to be matched

        # make the same shape as the output class, and points for masking
        positive_mask_expanded = positive_mask.unsqueeze(-1).repeat(1, 1, output_class.shape[2])
        # apply mask
        output_class_masked = output_class * positive_mask_expanded
        output_coord_masked = output_coord * positive_mask_expanded

        out = {"pred_logits": output_class_masked, "pred_points": output_coord_masked}

        return out


def build_mlp_two_stage(args, training):

    backbone = build_backbone(args)

    model = MLP_TwoStage(args.stage_one_model_path, backbone, args.num_classes, args.row, args.line)

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
        debris_class_idx=args.debris_class_idx,
        debris_radius=args.debris_radius,
        neg_lambda_debris=args.neg_lambda_debris,
        neg_lambda_other=args.neg_lambda_other,
        normalized_ce=args.normalized_ce,
    )

    return model, criterion