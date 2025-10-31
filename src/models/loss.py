import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.util.misc import (NestedTensor, accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized,
                       nested_tensor_from_tensor_list)


"""
Create objective function for the P2P pipeline, both classification and regression
    1. (LOSS) CE Loss
            loss term to optimize classification logits of each point proposal
    2. (LOSS) Regresssion Loss (L2 Distance)
            loss term to optimize
    3. (LOSS) Density Map Loss
            minimize density maps between prediction and gt (initializes guass kernals at 
            each point position)
    4. (REGULARIZATION) Count
            creats a course density map of the prediction, each grid represents number of
            points in the corresponding image patch, minimizes distance between gt and propsal maps
    5. (REGULARIZATION) Point proposal distance regulatization term
            penalizes points that are too close together, ad hoc way of minimizing double counting.

"""
class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, 
                    matcher, weight_dict, eos_coef, ce_coef, map_res, gauss_kernel_res, 
                    losses, debris_class_idx=None, debris_radius=None, neg_lambda_debris=None, neg_lambda_other=None):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            ce_coef: list of weight os each class in cross entropy loss (focal loss formulation)
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            debris_class_idx: class index of the debris class. If None, no debris class is used.
            debris_radius: radius of the debris class. If None, no debris class is used.
            neg_lambda_debris: weight of the debris class. If None, no debris class is used.
            neg_lambda_other: weight of the other class. If None, no other class is used.
        """
        
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher

        # --- initialize loss parameters ---
        self.losses = losses    # list of all the losses to be applied. See get_loss for list of available losses.
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.ce_coef = ce_coef
        self.map_res = map_res

        # --- initialize cross entropy focal loss weights ---
        ce_weight = torch.ones(self.num_classes + 1)
        ce_weight[0] = self.eos_coef
        for i, weight in enumerate(self.ce_coef):
            ce_weight[i + 1] = weight
        self.register_buffer("ce_weight", ce_weight)

        # --- initialize gaussian kernal for density map loss ---
        self.size = gauss_kernel_res
        if self.size % 2 == 0:
            raise(ValueError("map res must be odd"))
         # Calculate the range of x and y values
        ax = np.linspace(-(self.size // 2), self.size // 2, self.size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(5))
        self.kernel = np.outer(gauss, gauss)
        # Calculate the 2D Gaussian function
        self.kernel = self.kernel / np.sum(self.kernel)

        # --- initialize auxillary loss parameters ---
        self.aux_number = [2, 2]
        self.aux_range = [2, 8]
        self.aux_kwargs = {'pos_coef': 1., 'neg_coef': 1., 'pos_loc': 0.0002, 'neg_loc': 0.0002} 
            

        # --- initialize debris loss parameters ---
        self.debris_class_idx = debris_class_idx
        self.debris_radius = debris_radius
        self.neg_lambda_debris = neg_lambda_debris
        self.neg_lambda_other = neg_lambda_other

    def loss_labels(self, outputs, targets, indices, num_points, samples):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs and "pred_points" in outputs
        
        src_logits = outputs["pred_logits"] # --> [B,Q,C]
        anchor_points = outputs["pred_points"] # --> [B,Q,2]
        
        # --- extract target classes and points ---
        
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        
        # --- calcualte CE loss for each anchor points ---
        per_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, reduction="none")
        
        pos_mask = target_classes > 0
        neg_mask = ~pos_mask

        # --- identify debris-negative anchors by proximity to debris GT points ---
        
        debris_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        if self.debris_class_idx is not None:
            radius2 = self.debris_radius * self.debris_radius
            B, Q, _ = anchor_points.shape
            for b in range(B):
                # collect debris GT points for this sample
                t_labels = targets[b]["labels"]
                t_points = targets[b]["point"]
                if (t_labels == self.debris_class_idx).any():
                    dpts = t_points[(t_labels == self.debris_class_idx)].to(anchor_points.device).float()  # [D,2]
                    ap = anchor_points[b].float()  # [Q,2]
                    if dpts.numel() > 0:
                        # compute min squared distance from each anchor to any debris point
                        # ap: [Q,2], dpts: [D,2] => (Q,D,2) -> (Q,D) -> (Q,)
                        diff = ap[:, None, :] - dpts[None, :, :]
                        dist2 = (diff * diff).sum(-1)
                        min_dist2 = dist2.min(dim=1).values
                        debris_mask[b] = (min_dist2 <= radius2)

            other_neg_mask = neg_mask & ~debris_mask


        # Safe means per group
        def safe_mean(x): 
            return x.sum() / (x.numel() + 1e-9)

        loss_pos = safe_mean(per_ce[pos_mask]) if pos_mask.any() else per_ce.new_tensor(0.0)
        loss_neg_other = safe_mean(per_ce[other_neg_mask]) if other_neg_mask.any() else per_ce.new_tensor(0.0)

        if self.debris_class_idx is not None:
            loss_neg_debris = safe_mean(per_ce[debris_mask]) if debris_mask.any() else per_ce.new_tensor(0.0)
            loss_ce = loss_pos + self.neg_lambda_debris * loss_neg_debris + self.neg_lambda_other * loss_neg_other
        else:
            loss_ce = loss_pos + self.neg_lambda_other * loss_neg_other

        # --- calculate classwise loss --> useful for debugging and hyperparameter tuning ---
        classwise_loss_ce = []
        for i in range(self.num_classes + 1):
            weight = torch.zeros(self.num_classes + 1, device=src_logits.device)
            weight[i] = 1
            lce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight)
            classwise_loss_ce.append(lce)

        # calculate loss    
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.ce_weight
        )

        losses = {"loss_ce": loss_ce}
        class_losses = {"class_loss_ce": classwise_loss_ce}
        return losses, class_losses

    def loss_points(self, outputs, targets, indices, num_points, samples):
        '''
        Calculate L2 distance between matched points
        '''

        assert "pred_points" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        # point label
        target_classes = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )

        loss_bbox = F.mse_loss(src_points, target_points, reduction="none")

        # split points by class label, and recalculate mse by the specific class
        class_mse = []
        for i in range(self.num_classes):
            mask = (target_classes == i + 1).nonzero(as_tuple=True)
            src_point_mask = src_points[mask]
            target_point_mask = target_points[mask]
            eucdist = F.mse_loss(src_point_mask, target_point_mask, reduction="none")
            loss_avg = eucdist.sum() / int(mask[0].size()[0])
            class_mse.append(loss_avg)

        class_losses = {"class_loss_point": class_mse}
        losses = {}
        losses["loss_point"] = loss_bbox.sum() / num_points

        return losses, class_losses

    def loss_dense(self, outputs, targets, indices, num_points, samples):

        gaussian_kernel = self.kernel  # this kernel needs to be populated

        gt_heatmap = np.zeros([samples.size()[0], samples.size()[2], samples.size()[3]])
        target_points = [t["point"][i] for t, (_, i) in zip(targets, indices)]
        # populate the ground truth heatmap
        # loop over batch
        for i in range(samples.size()[0]):
            tar_points = target_points[i].detach().cpu()
            # loop over all target points in the sample
            for point in tar_points:
                # place guassian kernel at that poit
                for x in range(gaussian_kernel.shape[0]):
                    for y in range(gaussian_kernel.shape[1]):
                        x_coord = int(point[0]) - int((self.size - 1)/2) + x
                        y_coord = int(point[1]) - int((self.size - 1)/2) + y
                        if ((x_coord >= 0) and (x_coord < samples.size()[2])) and (
                            (y_coord >= 0) and (y_coord < samples.size()[3])
                        ):

                            gt_heatmap[i, x_coord, y_coord] += gaussian_kernel[x, y]

        # calculate point proposals
        pred_logits = outputs["pred_logits"].clone().detach().cpu()
        pred_points = outputs["pred_points"].clone().detach().cpu()
        prop_points = []
        for i in range(samples.size()[0]):
            outputs_scores = torch.nn.functional.softmax(
                pred_logits[i].unsqueeze(0), -1
            )[:, :, 1][0]
            points = (
                pred_points[i][outputs_scores > 0.5].detach().cpu().numpy().tolist()
            )
            prop_points.append(points)

        prop_heatmap = np.zeros(
            [samples.size()[0], samples.size()[2], samples.size()[3]]
        )
        # populate proposal density map
        for i in range(samples.size()[0]):
            for point in prop_points[i]:
                # place guassian kernel at that poit
                for x in range(gaussian_kernel.shape[0]):
                    for y in range(gaussian_kernel.shape[1]):
                        x_coord = int(point[0]) - int((self.size - 1)/2) + x
                        y_coord = int(point[1]) - int((self.size - 1)/2) + y
                        if ((x_coord >= 0) and (x_coord < samples.size()[2])) and (
                            (y_coord >= 0) and (y_coord < samples.size()[3])
                        ):

                            prop_heatmap[i, x_coord, y_coord] += gaussian_kernel[x, y]

        gt_heatmap = torch.Tensor(gt_heatmap)
        prop_heatmap = torch.Tensor(prop_heatmap)
        gt_heatmap = torch.flatten(gt_heatmap)
        prop_heatmap = torch.flatten(prop_heatmap)
        dist = (torch.sum(torch.abs(gt_heatmap - prop_heatmap))) / samples.size()[0]
        dist.requires_grad=True

        return {"loss_dense": dist}, {"class_loss_dense": 1.0}

    def loss_count(self, outputs, targets, indicies, num_points, samples):
        # create image paritions
        x_width = samples.size()[2] // self.map_res
        y_width = samples.size()[3] // self.map_res
        coords = []
        for i in range(4):
            for j in range(4):
                coords.append(
                    [
                        [i * (x_width), (i + 1) * (x_width)],
                        [j * (y_width), (j + 1) * (y_width)],
                    ]
                )
        pred_heatmap = np.zeros(shape=(samples.size()[0], self.map_res, self.map_res))
        gt_heatmap = np.zeros(shape=(samples.size()[0], self.map_res, self.map_res))

        # get prediction outputs and ground truth points
        assert "pred_points" in outputs
        assert "pred_logits" in outputs
        pred_points = outputs["pred_points"].clone().detach().cpu()
        pred_logits = outputs["pred_logits"].clone().detach().cpu()
        # populate the proposal heatmap estimation
        for batch in range(samples.size()[0]):
            outputs_scores = torch.nn.functional.softmax(
                pred_logits[batch].unsqueeze(0), -1
            )[:, :, 1][0]
            points = (
                pred_points[batch][outputs_scores > 0.5].detach().cpu().numpy().tolist()
            )
            points = torch.Tensor(points) 
            if points.size()[0] > 0:
                for i,current_coord in enumerate(coords):
                    # get indicies of all points proposal above treshold
                    idx = (
                        (points[:, 0] >= current_coord[1][0])
                         & (points[:, 0] < current_coord[1][1])
                         & (points[:, 1] >= current_coord[0][0])
                         & (points[:, 1] < current_coord[0][1])
                    )
                    num = torch.sum(idx == True)
                    pred_heatmap[batch][(i//self.map_res)][i - (self.map_res*(i//self.map_res))] = num.item() 
            
        # populate the ground truth heatmap
        for batch in range(len(targets)):
            points = targets[batch]["point"]
            if points.size()[0] > 0:
                for i,current_coord in enumerate(coords):
                    # get indicies of all points proposal above treshold
                    idx = (
                        (points[:, 0] >= current_coord[1][0])
                         & (points[:, 0] < current_coord[1][1])
                         & (points[:, 1] >= current_coord[0][0])
                         & (points[:, 1] < current_coord[0][1])
                    )
                    num = torch.sum(idx == True)
                    gt_heatmap[batch][i//self.map_res][i-(self.map_res*(i//self.map_res))] = num.item()
               
        difference_heatmap = gt_heatmap - pred_heatmap
        square_error = np.square(difference_heatmap)
        mean = np.mean(square_error)

        return {"loss_count": mean}, {"loss_count_classwise": 1.0}

    # self-regulation term to limit the distance between positive proposal points
    def loss_distance(self, outputs, targets, indicies, num_points, samples):
        assert "pred_logits" in outputs
        assert "pred_points" in outputs
        # return all positive samples
        pred_logits = outputs["pred_logits"].clone().detach().cpu()
        pred_points = outputs["pred_points"].clone().detach().cpu()

        positive_points = []
        for i in range(samples.size()[0]):
            outputs_scores = torch.nn.functional.softmax(
                pred_logits[i].unsqueeze(0), -1
            )[:, :, 1][0]
            points = (
                pred_points[i][outputs_scores > 0.5].detach().cpu().numpy().tolist()
            )
            positive_points.append(points)
        total_mean = []
        for i in range(len(positive_points)):
            prop_points = torch.Tensor(positive_points[i])
            if len(prop_points.size()) == 1:
                prop_points = torch.unsqueeze(prop_points, 0)
            # distance between each prop points
            dist = torch.cdist(
                torch.unsqueeze(prop_points, 0), torch.unsqueeze(prop_points, 0), p=2.0
            )
            if prop_points.size()[0] == 0 or prop_points.size()[0] == 1:
                topk = torch.full((1, 1, 2), float((2**32) - 1))
                vals = topk[:, :, 1:]
            elif prop_points.size()[0] < 4:
                topk = torch.topk(dist, 2, largest=False)
                vals = topk.values[:, :, 1:]
            else:
                topk = torch.topk(dist, 3, largest=False)
                vals = topk.values[:, :, 1:]

            # remove the first column of values (all zero, same points distance)
            mean = torch.mean(torch.mean(vals, dim=-1), dim=-1)
            total_mean.append(mean)
        mean = torch.mean(torch.Tensor(total_mean), dim=-1)
        return {"loss_distance": (1.0 / mean).to("cuda")}, {"class_loss_distance": 1.0}
    
    def loss_auxiliary(self, outputs, targets, indicies, num_points, samples):
        '''
        Auxillary points guidance, straight from the APGCC
        https://github.com/AaronCIH/APGCC/blob/main/apgcc/models/APGCC.py 
        '''
        # out: {"pred_logits", "pred_points", "offset"}
        # aux_out: {"pos0":out, "pos1":out, "neg0":out, "neg1":out, ...}
        loss_aux_pos = 0.
        loss_aux_neg = 0.
        loss_aux = 0.
        for n_pos in range(self.aux_number[0]):
            src_outputs = outputs['pos%d'%n_pos]
            # cls loss
            pred_logits = src_outputs['pred_logits'] # size=[1, # of gt anchors, 2] [p0, p1]
            target_classes = torch.ones(pred_logits.shape[:2], dtype=torch.int64, device=pred_logits.device) # [1, # of gt anchors], all sample is the head class
            loss_ce_pos = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
            # loc loss
            pred_points = src_outputs['pred_points'][0]
            target_points = torch.cat([t['point'] for t in targets], dim=0)
            target_points = target_points.repeat(1, int(pred_points.shape[0]/target_points.shape[0]))
            target_points = target_points.reshape(-1, 2)
            loss_loc_pos = F.mse_loss(pred_points, target_points, reduction='none')
            loss_loc_pos = loss_loc_pos.sum() / pred_points.shape[0]
            loss_aux_pos += loss_ce_pos + self.aux_kwargs['pos_loc'] * loss_loc_pos
        loss_aux_pos /= (self.aux_number[0] + 1e-9)

        for n_neg in range(self.aux_number[1]):
            src_outputs = outputs['neg%d'%n_neg]
            # cls loss
            pred_logits = src_outputs['pred_logits'] # size=[1, # of gt anchors, 2] [p0, p1]
            target_classes = torch.zeros(pred_logits.shape[:2], dtype=torch.int64, device=pred_logits.device) # [1, # of gt anchors], all sample is the head class
            loss_ce_neg = F.cross_entropy(pred_logits.transpose(1, 2), target_classes)
            # loc loss
            pred_points = src_outputs['offset'][0]
            target_points = torch.zeros(pred_points.shape, dtype=torch.float, device=pred_logits.device)
            loss_loc_neg = F.mse_loss(pred_points, target_points, reduction='none')
            loss_loc_neg = loss_loc_neg.sum() / pred_points.shape[0]
            loss_aux_neg += loss_ce_neg + self.aux_kwargs['neg_loc'] * loss_loc_neg
        loss_aux_neg /= (self.aux_number[1] + 1e-9)
        
        # show the output for the program; for debugging purposes
        if True:
            if self.aux_number[0] > 0:
                print("Auxiliary Training: [Pos] loss_cls:", loss_ce_pos, " loss_loc:", loss_loc_pos, " loss:", loss_aux_pos)
            if self.aux_number[1] > 0:
                print("Auxiliary Training: [Neg] loss_cls:", loss_ce_neg, " loss_loc:", loss_loc_neg, " loss:", loss_aux_neg)

        loss_aux = self.aux_kwargs['pos_coef']*loss_aux_pos + self.aux_kwargs['neg_coef']*loss_aux_neg
        losses = {'loss_aux': loss_aux}
        return losses, None     # TODO: Implement classiwse loss for the auxillary formulation

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_points, samples, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "points": self.loss_points,
            "density": self.loss_dense,
            "count": self.loss_count,
            "distance": self.loss_distance,
            "aux": self.loss_auxiliary,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_points, samples, **kwargs)

    def forward(self, outputs, targets, samples):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) = batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output = {
            "pred_logits": outputs["pred_logits"],
            "pred_points": outputs["pred_points"],
        }

        indices = self.matcher(output, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor(
            [num_points], dtype=torch.float, device=next(iter(output.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        classwise_losses = {}
        for loss in self.losses:
            main, classwise = self.get_loss(
                loss, output, targets, indices, num_boxes, samples
            )
            losses.update(main)
            classwise_losses.update(classwise)

        return losses, classwise_losses
