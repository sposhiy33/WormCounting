# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
Mostly copy-paste from DETR (https://github.com/facebookresearch/detr).
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from util.misc import NestedTensor
import numpy as np
import time
import torchvision.transforms as standard_transforms
import cv2

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def vis(samples, targets, pred, vis_dir, class_labels=None, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    class_labels -> list: [num_preds] -- predicited class of each predicted point
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for i,p in enumerate(pred[idx]):
            if class_labels is not None:
                if class_labels[i] == 0:
                    sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (225, 0, 255), -1)
                elif class_labels[i] == 1:
                    sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (255, 0, 0), -1)
                elif class_labels[i] == 2:
                    sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)
                else: pass
            else:
                sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name), 
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)


# the training routine for point proposal classification
def train_one_epoch_classifier(regr_model: torch.nn.Module,
                               model: torch.nn.Module, criterion: torch.nn.Module,
                               data_loader, optimizer, device, epoch, max_norm:float=0):
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
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    classwise_loss_total_epoch = []
    
    # iterate all training samples
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # forward
        outputs = model(samples)
        # calc the losses
        loss_dict, classwise_loss_dict = criterion(outputs, targets)
        classwise_loss_total_epoch.append(classwise_loss_dict)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce all losses
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
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
        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, classwise_loss_total_epoch


@torch.no_grad()
def evaluate_crowd_w_fine_grained(regr_model, class_model, data_loader, device, vis_dir="./visres"):
    """ 
    Compute regression points and corresponding classes based off regression to classification framework 
    Parameters:
    regr_model: P2P model --> nn.Module
    class_model: classification model --> nn.Module
    data_loader: validation dataloader --> utils.DataLoader
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    regr_model.eval()
    class_model.eval()
    regr_model.to(device)
    class_model.to(device)
    
    l1_count = 0 
    adult_count = 0 
    gt_l1_count = 0
    gt_adult_count = 0

    for samples,targets in data_loader:
        samples = samples.to(device)
        # two forward passes
        regression_outputs = regr_model(samples)
        classification_outputs = class_model(samples)

        regression_points = regression_outputs['pred_points'][0]
        regression_scores = regression_outputs['pred_logits']
        regression_scores = torch.nn.functional.softmax(regression_scores, -1)[:, :, 1][0]
        classification_score = classification_outputs['pred_logits'][0]
      
        classification_score = torch.nn.functional.softmax(classification_score, -1)
        gt_cnt = targets[0]['point'].shape[0]
        # regression threshold
        threshold = 0.55
        
        # pick out point proposals
        points = regression_points[regression_scores > threshold].detach().cpu().numpy().tolist()
        pred_cnt = int((regression_scores > threshold).sum())

        # point logits of regressed points 
        class_logits = classification_score[regression_scores > threshold].detach().cpu().numpy().tolist()
        class_labels = torch.zeros([len(class_logits)])
        for i,logit in enumerate(class_logits):
            # ignore the no-worm class
            logit[0] = 0
            class_labels[i] = logit.index(max(logit))
        class_labels = class_labels.numpy().tolist()
        targets_labels = targets[0]["labels"].detach().numpy().tolist()
        
        # get L1 CounGts
        l1_count += len([i for i in class_labels if i==1])
        adult_count += len([i for i in class_labels if i==2])
        gt_l1_count += len([i for i in targets_labels if i==1])
        gt_adult_count += len([i for i in targets_labels if i==2])


        if vis_dir is not None:
            vis(samples, targets, [points], vis_dir, class_labels=class_labels)

    print(l1_count, adult_count, gt_l1_count, gt_adult_count)

# the inference routine for p2p net
@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir="./visres", multiclass=False, num_classes=None):
    """ Evaluation script to evaluate models
    Parameters:
    model: torch.nn.Module 
    data_loader: torch.utils.data.DataLoader
    device: torch.device
    vis_dir: str path --> path where to save visualization
    multiclass: bool --> variable that enables the multiclass framework
    num_classes:
    """

    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []

    count = []
    gt_count = []

    for samples, targets in data_loader:
        samples = samples.to(device)

        outputs = model(samples)

        # logits and class_labels of point proposals, to be populated
        points = []
        class_labels = []

        outputs_points = outputs['pred_points'][0]
        target_labels = targets[0]['labels'].detach().numpy().tolist()
        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5
  
        predict_cnt = 0

        if multiclass:
            # iterate over each of the classes
            target_count = []
            proposal_count = [] 
            for i in range(num_classes): 
                class_idx = i + 1
                outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, class_idx][0]
                prop_points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
                for p in prop_points:
                    points.append(p)
                    class_labels.append(class_idx)
                cnt = int((outputs_scores > threshold).sum())
                predict_cnt += cnt
                proposal_count.append(cnt)
                target_count.append(len([i for i in target_labels if i == class_idx]))
            count.append(proposal_count)
            gt_count.append(target_count)
        else:
            outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
            points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
            cnt = len(points)        
            predict_cnt += cnt
            gt_count.append([len(target_labels)])
            count.append([cnt])

        # if specified, save the visualized images
        if vis_dir is not None: 
            if multiclass: vis(samples, targets, [points], vis_dir, class_labels)
            else: vis(samples, targets, [points], vis_dir)
        # accumulate MAE, MSE
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))

    gt_count = np.array(gt_count)
    count = np.array(count)
    final_gt_count = []
    final_count = []
    for i in range(gt_count.shape[1]):
        final_gt_count.append(np.sum(gt_count[:,i]))
        final_count.append(np.sum(count[:,i]))
    # calculate total
    print(f"gt_count: {final_gt_count}    count: {final_count}")
    return mae, mse
