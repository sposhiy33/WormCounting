# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
import time
from typing import Iterable
from collections import defaultdict

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms

import src.util.misc as utils


# the training routine for point proposal classification
def train_one_epoch_classifier(
    regr_model: torch.nn.Module,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader,
    optimizer,
    device,
    epoch,
    max_norm: float = 0,
):
    regr_model.eval()
    model.train()
    criterion.train()

    loss = []

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # get point proposals
        regr_points = regr_model(samples)

        outputs = model(samples)
        loss_dict = criterion(outputs, regr_points, targets)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # backward pass
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        loss.append(losses)

    loss = torch.stack(loss)
    avg_ce_loss = torch.mean(loss)
    return {"loss_ce": avg_ce_loss}


# the training routine
def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    classwise_loss_total_epoch = defaultdict(list)

    # iterate all training samples
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict, classwise_loss_dict = criterion(outputs, targets, samples)
        
        
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        
        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
 
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        # backward
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # update logger
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
        # collect classwise losses, does not support distributed training
        for k, v in classwise_loss_dict.items():
            classwise_loss_total_epoch[k].append(v)

    # reduce classwise losses
    averaged_classwise_losses = {}
    for loss_type, batch_losses in classwise_loss_total_epoch.items():
        if not batch_losses:
            continue
        
        # Get number of classes from first batch
        num_classes = len(batch_losses[0])
        averaged_per_class = []
        
        # For each class, collect losses across all batches and average
        for class_idx in range(num_classes):
            # Collect losses for this class from all batches
            class_losses = []
            for batch_loss in batch_losses:
                loss_tensor = batch_loss[class_idx]
                # Convert tensor to numpy and handle NaN values
                if torch.isnan(loss_tensor):
                    continue  # Skip NaN values
                class_losses.append(loss_tensor.item())
            
            # Compute average (only if we have valid values)
            if class_losses:
                avg_loss = np.mean(class_losses)
            else:
                avg_loss = np.nan  # If all were NaN, keep as NaN
            averaged_per_class.append(avg_loss)
        
        averaged_classwise_losses[loss_type] = averaged_per_class

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    }, averaged_classwise_losses


