'''
Classification P2P net with confusion matrix regularization
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
from .classification import *
from .mlp import *

class ConfusionClassifier(nn.Module): 
    '''
    Confusion classification (TP/FP) model using grid based label matching (MLP Classifer based)
    '''
    def __init__(self, backbone, num_classes, row=2, line=2):
        super().__init__()

        self.vgg_backbone = backbone
        self.num_classes = num_classes + 1   # classes for confusion classification (TN, TP, FN, FP)
        self.row = row
        self.line = line


        num_anchor_points = row * line

        self.linear = Linear(
            in_feat=256,
            out_feat=self.num_classes,
        )

        self.confusion_predictior = Linear(
            in_feat=256,
            out_feat=4,
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

        # pass through the classifer (for predicition) 
        classification = self.linear(features_fpn[1])
        
        # pass through the confusion classifier (for confusion classification)
        confusion = self.confusion_predictior(features_fpn[1])

        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        
        output_coord = anchor_points
        output_class = classification
        out = {"pred_logits": output_class, "pred_points": output_coord, "pred_confusion": confusion}

        return out


class Criterion_ConfusionClassification(nn.Module):
    '''
    Criterion for the confusion classification task (extension of vanilla Ture/False classification)
    '''
    def __init__(self, num_classes, matcher):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
    
    def loss_labels(self, outputs, targets, indices):
    
        pred_logits = outputs['pred_logits']
        pred_confusion = outputs['pred_confusion'] 
 
        # create ground truth grid,
        confusion_matrix = torch.full(
            (pred_logits.size(0), pred_logits.size(1)),
            0,
            device=pred_logits.device,
        )

        # get the T/F ground truth (from the matcher)
        idx = self._get_src_permutation_idx(indices)
        target_list = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device)
        
        target_classes[idx] = target_list

        # assign to each proposal point:
        # 0 - TN , 1- TP ,  2 - FN , 3 - FP 
        for batch in range(confusion_matrix.size(0)):
            for i in range(confusion_matrix.size(1)):
                # if the target class is 0 (TN) and the src class is 0 (TN) 
                ground = target_classes[batch, i]
                src_pred = torch.argmax(pred_logits[batch, i]).item()

                if (ground == 1) and (src_pred == 1):
                    confusion_matrix[batch, i] = 1
                elif (ground == 1) and (src_pred == 0):
                    confusion_matrix[batch, i] = 2
                elif (ground == 0) and (src_pred == 1):
                    confusion_matrix[batch, i] = 3
        
       # calculate loss    
        loss_ce = F.cross_entropy(
            pred_confusion.transpose(1, 2), confusion_matrix , self.ce_weight
        )

        return {"loss_ce": loss_ce}
          
    def get_loss(self, loss, outputs, targets, indices, num_points, samples, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_points, samples, **kwargs)

    def forward(self, outputs, targets, samples):
        '''
        foward pass on the loss computation
        '''
        output = {
            "pred_logits": outputs["pred_logits"],
            "pred_points": outputs["pred_points"],
        }

        # match the targets to the anchor points (equivalent of using output points and matching only using point distance)
        gt_indicies = self.matcher(output, targets, pointmatch=True) 
        # create the ground truth matrix
        gt_matrix = self.create_matrix(outputs, targets)
        # compute the loss
        loss_labels = self.loss_labels(outputs, targets, gt_indicies)

        # turn loss_labels
        #: implement classwise confusion loss
        return loss_labels, None

def build_confusion(args, training):

    backbone = build_backbone(args)
    model = ConfusionClassifier(backbone, args.num_classes, args.row, args.line)
    
    if not training:args.label_loss_coef,

    weight_dict = {
        "loss_ce": args.label_loss_coef,
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
 