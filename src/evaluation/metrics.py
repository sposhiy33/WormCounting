import torch
import numpy as np

from scipy.optimize import linear_sum_assignment

from src.evaluation.vis import *

"""
evaluation helper functions

TODO: allow for multiclass evaluation
"""

@torch.no_grad()
def get_output_points(model, sample, device, class_ind=1, threshold=0.5):

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
            
            points = get_output_points(model, samples, device)

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

    # Handle empty dataloader
    if len(error) == 0:
        print("WARNING: No samples processed in metric_mae_mse. Dataloader is empty.")
        return 0.0, 0.0

    # TODO: provide classwise validation errors.
    print(f'COUNTS (GT, PRED): {[[g,p] for g,p in zip(gt_cnts, pred_cnts)]}')
    print(f'Error: {error}')
    print(f'Square Error: {sq_error}')

    # calculate MAE, MSE
    mae = np.mean(error)
    mse = np.sqrt(np.mean(sq_error))

    return mae, mse

# # get the true positive points
def get_tp(points, targets, threshold:float, class_ind=1):

    if len(points) == 0:
        return 0,0,0

    points = torch.tensor(points)
    targets = targets["point"]

    # filter the points by class index

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
            points = get_output_points(model, samples, device)

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
    model, data_loader, device, vis_dir=None):
    model.eval()

    # calculate the mean absolute error and mean square error
    mae,mse = metric_mae_mse(model, data_loader, device=device, vis_dir=vis_dir)

    # calculate localization metrics
    loc_dict = {}
    for val in [4,8]:
        prec, rec, f1 = metric_precision_recall_f1(model, data_loader, device=device, threshold=val)
        loc_dict[f"{val}"] = (prec, rec, f1)

    return mae, mse, loc_dict
