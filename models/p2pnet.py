"""
Implementation of vanilla P2P Net
Implements model architecutre and loss formulation
Introduced some archtecutre changes and additional loss terms
"""


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


# the network frmawork of the regression branch
class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchor_points * 2, kernel_size=3, padding=1
        ) 

    # sub-branch forward
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.output(out)

        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 2)


# the network frmawork of the classification branch
class ClassificationModel(nn.Module):
    def __init__(
        self,
        num_features_in,
        num_anchor_points=4,
        num_classes=80,
        prior=0.01,
        feature_size=256,
    ):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(
            feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1
        )
        self.output_act = nn.Sigmoid()

    # sub-branch forward
    def forward(self, x):
        out_1 = self.conv1(x)
        out_1 = self.act1(out_1)

        out_2 = self.conv2(out_1)
        out_2 = self.act2(out_2)

        out = self.output(out_2)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, _ = out1.shape

        out2 = out1.view(
            batch_size, width, height, self.num_anchor_points, self.num_classes
        )
        import pdb; pdb.set_trace()
        return out2.contiguous().view(x.shape[0], -1, s048-elf.num_classes)


# generate the reference points in grid layout
def generate_anchor_points(stride=16, row=3, line=3):
    row_step = stride / row
    line_step = stride / line

    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    anchor_points = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    return anchor_points


# shift the meta-anchor to get an acnhor points
def shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchor_points = anchor_points.reshape((1, A, 2)) + shifts.reshape(
        (1, K, 2)
    ).transpose((1, 0, 2))
    all_anchor_points = all_anchor_points.reshape((K * A, 2))

    return all_anchor_points


# this class generate all reference points on all pyramid levels
class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, strides=None, row=3, line=3):
        super(AnchorPoints, self).__init__()

        if pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        else:
            self.pyramid_levels = pyramid_levels

        if strides is None:
            self.strides = [2**x for x in self.pyramid_levels]

        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = image.shape[2:]
        image_shape = np.array(image_shape)
        image_shapes = [(image_shape + 2**x - 1) // (2**x) for x in self.pyramid_levels]

        all_anchor_points = np.zeros((0, 2)).astype(np.float32)
        # get reference points for each level
        for idx, p in enumerate(self.pyramid_levels):
            anchor_points = generate_anchor_points(2**p, row=self.row, line=self.line)
            shifted_anchor_points = shift(
                image_shapes[idx], self.strides[idx], anchor_points
            )
            all_anchor_points = np.append(
                all_anchor_points, shifted_anchor_points, axis=0
            )

        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        # send reference points to device
        if torch.cuda.is_available():
            return torch.from_numpy(all_anchor_points.astype(np.float32)).cuda()
        else:
            return torch.from_numpy(all_anchor_points.astype(np.float32))


class Decoder(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(Decoder, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P3_2 = nn.Conv2d(
            feature_size, feature_size, kernel_size=3, stride=1, padding=1
        )

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


# the defenition of the P2PNet model
class P2PNet(nn.Module):
    def __init__(self, backbone, num_classes, noreg=False, row=2, line=2):
        super().__init__()
        self.noreg = noreg

        self.backbone = backbone
        # number of classes in each logit, add one for no-person class.
        self.num_classes = num_classes + 1
        # the number of all anchor points
        num_anchor_points = row * line

        self.regression = RegressionModel(
            num_features_in=256, num_anchor_points=num_anchor_points
        )
        self.classification = ClassificationModel(
            num_features_in=256,
            num_classes=self.num_classes,
            num_anchor_points=num_anchor_points,
        )

        self.anchor_points = AnchorPoints(
            pyramid_levels=[
                2,
            ],
            row=row,
            line=line,
        )

        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples: NestedTensor):
        # get the backbone features
        features = self.backbone(samples)
        # forward the feature pyramid
        features_fpn = self.fpn([features[1], features[2], features[3]])
        batch_size = features[0].shape[0]
        # run the regression and classification branch
        regression = self.regression(features_fpn[0]) * 100  # 8x
        classification = self.classification(features_fpn[0])
        anchor_points = self.anchor_points(samples).repeat(batch_size, 1, 1)
        # decode the points as prediction
        if self.noreg==True: 
            output_coord = anchor_points
        else: 
            output_coord = regression + anchor_points
        output_class = classification
        out = {"pred_logits": output_class, "pred_points": output_coord}

        # set trace
        # import pdb; pdb.set_trace()

        return out


class FineClassifier(nn.Module):
    """Create fine grained classifier for p2p net point proposals"""

    def __init__(self, backbone, num_classes, row=2, line=2):
        super().__init__()

        self.vgg_backbone = backbone
        self.num_classes = num_classes
        self.row = row
        self.line = line

        num_anchor_points = row * line

        self.classification = ClassificationModel(
            num_features_in=256,
            num_classes=self.num_classes,
            num_anchor_points=num_anchor_points,
        )
        self.fpn = Decoder(256, 512, 512)

    def forward(self, samples: NestedTensor):
        # get the backbone (vgg) features
        features = self.vgg_backbone(samples)
        # construct the feature space
        features_fpn = self.fpn([features[1], features[2], features[3]])
        batch_size = features[0].shape[0]

        # pass through the classifer
        classification = self.classification(features_fpn[1])
        output_class = classification

        out = {"pred_logits": output_class}

        return out


"""
Create objective function that does classification only from the grid of points
"""


class SetCriterion_Classification(nn.Module):

    def __init__(self, matcher, num_classes, eos_coef, ce_coef):
        super().__init__()
        self.matcher = matcher  # this is the same matcher used in the P2P step
        self.num_classes = num_classes
        empty_weight = torch.ones(num_classes)
        empty_weight[0] = eos_coef
        for i, weight in enumerate(ce_coef):
            empty_weight[i + 1] = weight
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_points):
        """CE Loss: between each point proposal and corresponding ground truth point"""
        # calculate the point proposals from each input patch
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        losses = {"loss_ce": loss_ce}

        return losses

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

    def forward(self, outputs, regression_outputs, targets):
        """loss computation"""
        regression_outputs = {
            "pred_logits": regression_outputs["pred_logits"],
            "pred_points": regression_outputs["pred_points"],
        }
        indicies = self.matcher(regression_outputs, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor(
            [num_points],
            dtype=torch.float,
            device=next(iter(regression_outputs.values())).device,
        )

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indicies, num_points))

        return losses


"""
Create objective function for the P2P pipeline, both classification and regression
"""


class SetCriterion_Crowd(nn.Module):

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, ce_coef, map_res, gauss_kernel_res, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.ce_coef = ce_coef
        self.map_res = map_res
        self.losses = losses
        ce_weight = torch.ones(self.num_classes + 1)
        ce_weight[0] = self.eos_coef
        for i, weight in enumerate(self.ce_coef):
            ce_weight[i + 1] = weight
        self.register_buffer("ce_weight", ce_weight)

        # initialize gaussian kernal
        self.size = gauss_kernel_res
        if self.size % 2 == 0:
            raise(ValueError("map res must be odd"))
         # Calculate the range of x and y values
        ax = np.linspace(-(self.size // 2), self.size // 2, self.size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(5))
        self.kernel = np.outer(gauss, gauss)
        # Calculate the 2D Gaussian function
        self.kernel = self.kernel / np.sum(self.kernel)
            
    def loss_labels(self, outputs, targets, indices, num_points, samples):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )

        target_classes[idx] = target_classes_o

        ## classwise loss for debugging
        classwise_loss_ce = []
        for i in range(self.num_classes + 1):
            weight = torch.zeros(self.num_classes + 1, device=src_logits.device)
            weight[i] = 1
            lce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, weight)
            classwise_loss_ce.append(lce)

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.ce_weight
        )

        losses = {"loss_ce": loss_ce}
        class_losses = {"class_loss_ce": classwise_loss_ce}

        return losses, class_losses

    def loss_points(self, outputs, targets, indices, num_points, samples):

        assert "pred_points" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_points = outputs["pred_points"][idx]
        target_points = torch.cat(
            [t["point"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        # point labels
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
        dist = (torch.sum(torch.abs(gt_heatmap - prop_heatmap)).item()) / samples.size()[0]

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
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_points, samples, **kwargs)

    def forward(self, outputs, targets, samples):
        """This performs the loss computation.
        print(pred_logits.size())
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        output1 = {
            "pred_logits": outputs["pred_logits"],
            "pred_points": outputs["pred_points"],
        }

        indices1 = self.matcher(output1, targets)

        num_points = sum(len(t["labels"]) for t in targets)
        num_points = torch.as_tensor(
            [num_points], dtype=torch.float, device=next(iter(output1.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_boxes = torch.clamp(num_points / get_world_size(), min=1).item()

        losses = {}
        classwise_losses = {}
        for loss in self.losses:
            main, classwise = self.get_loss(
                loss, output1, targets, indices1, num_boxes, samples
            )
            losses.update(main)
            classwise_losses.update(classwise)

        return losses, classwise_losses


# create the P2PNet model
def build_p2p(args, training):
    # treats persons as a single class

    backbone = build_backbone(args)
    model = P2PNet(backbone, args.num_classes, args.noreg, args.row, args.line)
    if not training:
        return model

    weight_dict = {
        "loss_ce": 1,
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

## 
def build_multiclass(args, training):

    backbone = build_backbone(args)
    model = FineClassifier(backbone, args.downstream_num_classes, args.row, args.line)
    if not training:
        return model

    # build the matcher based on original P2P model
    matcher = build_matcher_crowd(args, override_multiclass=True)
    criterion = SetCriterion_Classification(
        matcher=matcher,
        num_classes=args.downstream_num_classes,
        eos_coef=args.eos_coef,
        ce_coef=args.ce_coef,
    )

    return model, criterion
