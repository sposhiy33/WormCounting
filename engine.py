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

from scipy.optimize import linear_sum_assignment

import cv2
import numpy as np
import torch
import torchvision.transforms as standard_transforms

import util.misc as utils
from util.misc import NestedTensor


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def vis(samples, pred, vis_dir, eval=False, targets=None, class_labels=None, img_name=None):
    """
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    class_labels -> list: [num_preds] -- predicited class of each predicted point
    """
    if samples.ndim < 4:
        samples = samples.unsqueeze(0)

    try: gts = [targets["point"].tolist()]
    except: print("targets not specified, performing pure eval") 

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose(
        [
            DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            standard_transforms.ToPILImage(),
        ]
    )
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert("RGB")).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 5

        # Generate and save prediction images
        for i, p in enumerate(pred[idx]):
            if class_labels is not None:
                if class_labels[i] == 0:
                    sample_pred = cv2.circle(
                        sample_pred, (int(p[0]), int(p[1])), size, (0, 255, 0), -1
                    )
                elif class_labels[i] == 1:
                    sample_pred = cv2.circle(
                        sample_pred, (int(p[0]), int(p[1])), size, (255, 0, 0), -1
                    )
                elif class_labels[i] == 2:
                    sample_pred = cv2.circle(
                        sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1
                    )
                else:
                    pass
            else:
                sample_pred = cv2.circle(
                    sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1
                )

        try: name = targets["image_id"]
        except: name = img_name

        if eval:
            file_name = "{}_{}_pred.jpg".format(int(name), len(pred[idx]))
        else: file_name = "{}_gt_{}_pred_{}_pred.jpg".format(int(name), len(gts[idx]), len(pred[idx])) 
        cv2.imwrite(
            os.path.join(
                vis_dir,
                file_name,
            ),
            sample_pred,
        )

        # if performing pure eval, do not continue
        # saving the groun truth images

        # draw gt
        if not eval:
            for t in gts[idx]:
                sample_gt = cv2.circle(
                    sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1
                )

            cv2.imwrite(
                os.path.join(
                    vis_dir,
                    "{}_gt_{}_pred_{}_gt.jpg".format(
                        int(name), len(gts[idx]), len(pred[idx])
                    ),
                ),
                sample_gt,
            )


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
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    classwise_loss_total_epoch = []

    # iterate all training samples
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict, classwise_loss_dict = criterion(outputs, targets, samples)
        classwise_loss_total_epoch.append(classwise_loss_dict)
        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        
        # print(f"loss dict: {loss_dict}")
        # print(f"weight dict: {weight_dict}")
        # print(f"Losses: {losses}")
        
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
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {
        k: meter.global_avg for k, meter in metric_logger.meters.items()
    }, classwise_loss_total_epoch


"""
evaluation helper functions

TODO: allow for multiclass evaluation
"""

@torch.no_grad()
def get_output_points(model, sample, target, device, class_ind=1, threshold=0.5):

    sample = sample.to(device)
    sample = sample.unsqueeze(0)
    outputs = model(sample)

    # to populate
    points = []
    class_labels = []

    outputs_points = outputs["pred_points"][0]

    outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, class_ind][0]
    points = (
        outputs_points[outputs_scores > threshold]
        .detach()
        .cpu()
        .numpy()
        .tolist()
    )

    return points
 
# mse and mae - count metrics
def metric_mae_mse(model, dataloader, device, vis_dir=None):

    error = []
    sq_error = []

    pred_cnts = []
    gt_cnts = []

    for batch, batch_targets in dataloader:
        for samples, targets in zip(batch, batch_targets):
            
            points = get_output_points(model, samples, targets, device)

            # get the predicted count and the ground truth count
            pred_cnt = len(points)
            gt_cnt = targets["point"].shape[0]
            
            pred_cnts.append(pred_cnt)
            gt_cnts.append(gt_cnt)

            # calculate the mean absolute error and mean square error
            mae = abs(pred_cnt - gt_cnt)
            mse = (pred_cnt - gt_cnt) * (pred_cnt - gt_cnt)

            error.append(float(mae))
            sq_error.append(float(mse))

            # if specified, save the visualized images
            if vis_dir is not None:
                vis(samples, [points], vis_dir, targets=targets)

    print(f'COUNTS (GT, PRED): {[[g,p] for g,p in zip(gt_cnts, pred_cnts)]}')
    print(f'Error: {error}')
    print(f'Square Error: {sq_error}')

    # calculate MAE, MSE
    mae = np.mean(error)
    mse = np.sqrt(np.mean(sq_error))

    return mae, mse

# # get the true positive points
def get_tp(points, targets, threshold:float):

    if len(points) == 0:
        return 0,0,0

    points = torch.tensor(points)
    targets = targets["point"]

    num_preds = len(points)
    num_gt = targets.shape[0]

    # calculate the distance between the points
    dist_matrix = torch.cdist(points, targets, p=2)
    # cost matrix
    C = dist_matrix.detach().cpu()

    pred_indices, gt_indices = linear_sum_assignment(C)
    
    tp_count = 0
    # Iterate through the optimal assignments
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        # If the distance for this assignment is within the threshold, it's a TP
        if dist_matrix[pred_idx, gt_idx] < threshold:
            tp_count += 1
            
    fp_count = num_preds - tp_count
    fn_count = num_gt - tp_count if (num_gt - tp_count) > 0 else 0 
    
    return tp_count, fp_count, fn_count
    
# precision, recall, f1 score - localization performance
def metric_precision_recall_f1(model, dataloader, threshold, device):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for batch, batch_targets in dataloader: 
        for samples, targets in zip(batch, batch_targets):
            points = get_output_points(model, samples, targets, device)

            pred_cnt = len(points)
            gt_cnt = targets["point"].shape[0]

            # calculate precision, recall, f1 score
            tp, fp, fn = get_tp(points, targets, threshold=threshold)

            total_tp += tp
            total_fp += fp
            total_fn += fn
            
    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    return prec, rec, f1 

# the inference routine for p2p net
# validation, error with ground truth points
@torch.no_grad()
def evaluate_crowd_no_overlap(
    model, data_loader, device, vis_dir=None, multiclass=None, num_classes=None):
    model.eval()
 
    mae,mse = metric_mae_mse(model, data_loader, device=device, vis_dir=vis_dir)


    # calculate localization metrics
    loc_dict = {}
    for val in [4,8]:
        prec, rec, f1 = metric_precision_recall_f1(model, data_loader, device=device, threshold=val)
        loc_dict[f"{val}"] = (prec, rec, f1)

    return mae, mse, loc_dict
