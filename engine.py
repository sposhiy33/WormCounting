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



@torch.no_grad()
def evaluate_crowd(model, data_loader, device, vis_dir=None):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.eval()
    model.to(device)
    
    for idx, batch in enumerate(data_loader):
        img = batch[0]
        img = img.to(device)
        outputs = model(img)
        # print(f"Pred Points: {outputs["pred_points"].size()}")
        # print(f"Pred Logits: {outputs["pred_logits"].size()}")
        # logits and class_labels of point proposals, to be populated
        points = []
        class_labels = []

        outputs_points = outputs["pred_points"][0]
       
        # 0.5 is used by default
        threshold = 0.5

        class_idx = 0 + 1
        outputs_scores = torch.nn.functional.softmax(
            outputs["pred_logits"], -1)[:, :, class_idx][0]
        prop_points = (
            outputs_points[outputs_scores > threshold]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        for p in prop_points:
            points.append(p)
            class_labels.append(class_idx)
        predict_cnt = int((outputs_scores > threshold).sum())

        print(predict_cnt)

        if vis_dir != None:
            vis(img, [points], vis_dir, eval=True, class_labels=class_labels, img_name=idx) 
                
        

# the inference routine for p2p net
# validation, error with ground truth points
@torch.no_grad()
def evaluate_crowd_no_overlap(
    model, data_loader, device, vis_dir=None, multiclass=None, num_classes=None
):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    # run inference on all images to calc MAE
    maes = []
    mses = []
    
    count = []
    gt_count = []

    class_mae = []
    class_mse = []
    class_dist = []

    # iterate over each of the classes
    # class wise metrics
    for i in range(num_classes):
        i_mae = []
        i_mse = []
        dist_list = []
        for batch, batch_targets in data_loader:
            for samples, targets in zip(batch, batch_targets):
                samples = samples.to(device)
                samples = samples.unsqueeze(0)
                # print(f"Samples: {samples.size()}")
                outputs = model(samples)
                # print(f"Pred Points: {outputs["pred_points"].size()}")
                # print(f"Pred Logits: {outputs["pred_logits"].size()}")
                # logits and class_labels of point proposals, to be populated
                points = []
                class_labels = []

                outputs_points = outputs["pred_points"][0]
                target_labels = targets["labels"].detach().numpy().tolist()
                target_point = targets["point"].detach().numpy().tolist()
               
                ground_truth_points = []
                # truth map --
                for index, class_type in enumerate(target_labels):
                    if class_type == i + 1:
                        ground_truth_points.append(target_point[index])


                # 0.5 is used by default
                threshold = 0.5
                predict_cnt = 0

                class_idx = i + 1

                outputs_scores = torch.nn.functional.softmax(
                    outputs["pred_logits"], -1)[:, :, class_idx][0]
                
                # final output points 
                prop_points = (
                    outputs_points[outputs_scores > threshold]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

                # calculate distance between each predicted point and the closest ground truth point 
                if (len(ground_truth_points) > 0) and (len(prop_points) > 0):
                    dist = torch.cdist(torch.Tensor(ground_truth_points),
                                   torch.Tensor(prop_points),
                                   p=2)
                    min_dist = torch.min(dist, 1)
                    mean_dist = torch.mean(min_dist[0])
                    dist_list.append(mean_dist.item())
                for p in prop_points:
                    points.append(p)
                    class_labels.append(class_idx)
                predict_cnt = int((outputs_scores > threshold).sum())
                gt_cnt = len([i for i in target_labels if i == class_idx])

                mae = abs(predict_cnt - gt_cnt)
                mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
                i_mae.append(mae)
                i_mse.append(mse)
        i_mae = sum(i_mae) / len(i_mae)
        i_mse = sum(i_mse) / len(i_mse)
        i_mse = math.sqrt(i_mse)
        if len(dist_list) > 0:
            dist_list = sum(dist_list) / len(dist_list)
        else: dist_list = math.inf 
        class_mae.append(i_mae)
        class_mse.append(i_mse)
        class_dist.append(dist_list)

    print(f"MAE: {class_mae}")
    print(f"MSE: {class_mse}")
    print(f"DIST: {class_dist}")

    # general eval pass, not class wise 
    for batch, batch_targets in data_loader:
        for samples, targets in zip(batch, batch_targets): 
            samples = samples.to(device)
            samples = samples.unsqueeze(0)
            # print(f"Samples: {samples.size()}")
            outputs = model(samples)
            # print(f"Pred Points: {outputs["pred_points"].size()}")
            # print(f"Pred Logits: {outputs["pred_logits"].size()}")
            # logits and class_labels of point proposals, to be populated
            points = []
            class_labels = []

            outputs_points = outputs["pred_points"][0]
            target_labels = targets["labels"].detach().numpy().tolist()
            gt_cnt = targets["point"].shape[0]
           
            # 0.5 is used by default
            threshold = 0.5

            predict_cnt = 0

            if len(multiclass)>0:
                # iterate over each of the classes
                target_count = []
                proposal_count = []
                for i in range(num_classes):
                    class_idx = i + 1
                    outputs_scores = torch.nn.functional.softmax(
                        outputs["pred_logits"], -1)[:, :, class_idx][0]
                    prop_points = (
                        outputs_points[outputs_scores > threshold]
                        .detach()
                        .cpu()
                        .numpy()
                        .tolist()
                    )
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
                outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
                points = (
                    outputs_points[outputs_scores > threshold]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )
                cnt = len(points)
                predict_cnt += cnt
                gt_count.append([len(target_labels)])
                count.append([cnt])

            # if specified, save the visualized images
            if vis_dir is not None:
                if multiclass:
                    vis(samples, [points], vis_dir, class_labels=class_labels, targets=targets)
                else:
                    vis(samples, [points], vis_dir, targets=targets)
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
    print(gt_count)
    print(count)
    for i in range(gt_count.shape[1]):
        final_gt_count.append(np.sum(gt_count[:, i]))
        final_count.append(np.sum(count[:, i]))
    # print counts
    print(f"gt_count: {final_gt_count}    count: {final_count}")
    return mae, mse

'''
def confusion(model, data_loader, matcher, multiclass=None, num_classes=None, device=None):
    
    model.eval()
    
    for batch, batch_targets in data_loader:
        for samples, targets in zip(batch, batch_targets): 
            
            samples = samples.to(device)
            samples = samples.unsqueeze(0)
            outputs = model(samples)
            # logits and class_labels of point proposals, to be populated
            points = []
            class_labels = []

            outputs_points = outputs["pred_points"][0]
            target_labels = targets["labels"].detach().numpy().tolist()
            gt_cnt = targets["point"].shape[0]

            # matcher
            indices = matcher(outputs, targets, pointmatch=True)

            # get the final output points
            # 0.5 is used by default
            threshold = 0.5

            # get the predicted points
            pred_logits = outputs["pred_logits"][0]    
            output_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
            points = (
                    outputs_points[output_scores > threshold]
                    .detach()
                    .cpu()
                    .numpy()
                    .tolist()
                )

            # create empty list for confusion labels
            confusion_matrix = torch.full(
                (1, output_scores[0].size())
                0, 
            )

            #  use indicies to get the correct 

            # assign to each proposal point:
            # 0 - TN , 1- TP ,  2 - FN , 3 - FP 
            for batch in range(confusion_matrix.size(0)):
                for i in range(confusion_matrix.size(1)):
                    # if the target class  is 0 (TN) and the src class is 0 (TN) 
                    ground = target_classes[batch, i]
                    src_pred = torch.argmax(pred_logits[batch, i]).item()

                    if (ground == 1) and (src_pred == 1):
                        confusion_matrix[batch, i] = 1
                    elif (ground == 1) and (src_pred == 0):
                        confusion_matrix[batch, i] = 2
                    elif (ground == 0) and (src_pred == 1):
                        confusion_matrix[batch, i] = 3
            
    


                        
            outputs_scores = torch.nn.functional.softmax(outputs["pred_logits"], -1)[:, :, 1][0]
            points = (
                outputs_points[outputs_scores > threshold]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            cnt = len(points)
            predict_cnt += cnt
            gt_count.append([len(target_labels)])
            count.append([cnt])
'''
 