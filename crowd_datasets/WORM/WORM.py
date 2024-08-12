import glob
import math
import os
import random

import cv2
import numpy as np
import scipy.io as io
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset


class WORM(Dataset):
    def __init__(
        self,
        data_root,
        transform=None,
        train=False,
        scale=False,
        rotate=False,
        patch=False,
        equal_crop=False,
        flip=False,
        multiclass=False,
        class_filter=None,
        hsv=False,
        hse=False,
        edges=False,
    ):
        self.root_path = data_root

        if "worm_dataset" in self.root_path:
            self.train_lists = "shtrain.list"
            self.eval_list = "shtest.list"
        else:
            self.train_lists = "mtrain.txt"
            self.eval_list = "mtest.txt"

        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(",")
        if train:
            self.img_list_file = self.train_lists.split(",")
        else:
            self.img_list_file = self.eval_list.split(",")

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = (
                        os.path.join(self.root_path, line[1].strip())
                    )
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.rotate = rotate
        self.train = train
        self.patch = patch
        self.scale = scale
        self.flip = flip
        self.multiclass = multiclass
        self.equal_crop = equal_crop
        self.class_filter = class_filter
        self.hsv = hsv
        self.hse = hse
        self.edges = edges

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), "index range error"

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point, labels = load_data(
            (img_path, gt_path), self.train, self.multiclass, self.class_filter
        )

        if self.edges:
            img = edges(img)

        # apply augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train and self.scale:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            print(scale)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(
                    img.unsqueeze(0), scale_factor=scale
                ).squeeze(0)
                point *= scale

        # crop augumentaiton
        if self.train and self.patch:
            img, point, labels = random_crop(img, point, labels)

            # convert point arrays for each image to torch Tensor type
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])
        elif self.equal_crop:
            img, point, labels = equal_crop(img,point,labels)
        else:
            img = img.detach().numpy()
            point = [point]
            labels = [labels]

        # a rotation augmentation applied at the patch level
        if self.train and self.rotate:
            # randomly rotate the image
            img, point, labels = random_rotate(img, point, labels, 4)

        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

        if self.hsv:
            img = rgb_to_hsv(img)

        if self.hse:
            try: img = img.detach().numpy()
            except: print("img is already a numpy array")
            img = rgb_to_hse(img)


        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]["point"] = torch.Tensor(point[i])
            image_id = int(img_path.split("/")[-1].split(".")[0].split("_")[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]["image_id"] = image_id
            target[i]["labels"] = torch.Tensor(labels[i].tolist()).long()

        return img, target


def load_data(img_gt_path, train, multiclass, class_filter):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []

    # assign a class
    labels = []
    with open(gt_path) as f_label:
        for line in f_label:
            if "\t" in line:
                elements = len(line.strip().split("\t"))
                x = float(line.strip().split("\t")[0].strip())
                y = float(line.strip().split("\t")[1].strip())
                points.append([x, y])
                # create labels
                if multiclass:
                    # if the label is included in the point txt file, use this scheme
                    if elements == 3:
                        lab = str(line.strip().split("\t")[2].strip())
                        if lab == "Gravid":
                            labels.append(2)
                        elif lab == "L1":
                            labels.append(1)
                    # else infer label from the img file name
                    else:
                        if "L1" in img_path:
                            labels.append(1)
                        elif "ADT" in img_path:
                            labels.append(2)
                        else:
                            labels.append(2)  # default to the adult label
                else:
                    labels.append(1)
            else:
                x = float(line.strip().split(" ")[0].replace(",", ""))
                y = float(line.strip().split(" ")[1])
                labels.append(2)
                points.append([x, y])

    if class_filter != None:

        class_filter_mask = []
        for i in labels:
            if i == class_filter: class_filter_mask.append(True)
            else: class_filter_mask.append(False)
        labels = [i for keep, i in zip(class_filter_mask, labels) if keep]
        points = [i for keep, i in zip(class_filter_mask, points) if keep]

    return img, np.array(points), np.array(labels)


def rgb_to_hsv(rgb):
    """
    Implementation taken from: https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
    Parameters:
    img --> torch.Tensor of shape
    """
    rgb = torch.Tensor(rgb)
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def rgb_to_hse(rgb):
    """
    Channels:
        1. Hue
        2. Saturation
        3. Naive edge detection filter
    """
    # generate edges
    # cv2.Canny() takes input of shape (H, W, channels)
    # output as (H, W)
    edges = []
    if len(rgb.shape) <= 3:
        rgb = np.expand_dims(rgb, axis=0)
    for i in range(rgb.shape[0]):
        rgb_e = (rgb[i] * 255).astype(np.uint8)
        rgb_e = np.transpose(rgb_e, (1, 2, 0))
        edge = cv2.Canny(rgb_e, 50, 150)
        edge = torch.Tensor(edge)
        edge = edge.unsqueeze(0).unsqueeze(0)
        edges.append(edge)
    # double unsqueeze to match output shape of H and S
    edges = torch.cat(edges, dim=0)

    # generate H and S
    # require input of size (batch_size, channels, H, W)
    rgb = torch.Tensor(rgb)
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.0
    hsv_h /= 6.0
    hsv_s = torch.where(cmax == 0, torch.tensor(0.0).type_as(rgb), delta / cmax)

    return torch.cat([hsv_h, hsv_s, edges], dim=1)


def edges(rgb):
    blurSize = 10
    mCannyMin = 40
    mCannyMax = 100
    mCircRadMin = 200
    mCircRadMax = 500
    radiusDecrease = 30
    circleOffset = -5

    clipLimit = 2.0
    tileGridSize = (8, 8)
    wCannyMin = 30
    wCannyMax = 120
    erosionK = 2
    dilationK = 2
    resizeWorm = False

    # PIL to numpy
    img_tensor = F.pil_to_tensor(rgb)
    rgb = img_tensor.detach().numpy()
    # convert to [0,255] range from PIL range
    image = (rgb * 255).astype(np.uint8)
    # change to H,W,C as needed by OpenCV
    image = np.transpose(image, (1, 2, 0))
    # generate masks
    petriMask = get_petri_mask(
        image,
        blurSize,
        mCannyMin,
        mCannyMax,
        mCircRadMin,
        mCircRadMax,
        radiusDecrease,
        circleOffset,
    )
    wormMask = get_worm_mask(
        image,
        tileGridSize,
        clipLimit,
        wCannyMin,
        wCannyMax,
        erosionK,
        dilationK,
        resizeWorm,
    )
    result = cv2.bitwise_and(petriMask, wormMask)
    # convert result to three color channels from one
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    result = Image.fromarray(result)
    return result


def get_petri_mask(
    image,
    blurSize,
    mCannyMin,
    mCannyMax,
    circRadMin,
    circRadMax,
    radiusDecrease,
    circleOffset,
):
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayImage = clahe.apply(grayImage)
    grayImage = cv2.blur(grayImage, (blurSize, blurSize))
    edges = cv2.Canny(grayImage, mCannyMin, mCannyMax)
    detected_circles = cv2.HoughCircles(
        edges,
        cv2.HOUGH_GRADIENT,
        1,
        20,
        param1=50,
        param2=30,
        minRadius=circRadMin,
        maxRadius=circRadMax,
    )
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        pt = detected_circles[0, 0, :]
        a, b, r = pt[0], pt[1], pt[2]
        mask = np.zeros_like(grayImage)
        mask = cv2.circle(
            mask,
            (a - circleOffset, b - circleOffset),
            r - radiusDecrease,
            (255, 255, 255),
            -1,
        )
        return mask


def get_worm_mask(
    image,
    tileGridSize,
    clipLimit,
    wCannyMin,
    wCannyMax,
    erosionK,
    dilationK,
    resizeWorm,
):
    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    grayImage = clahe.apply(grayImage)
    # gray_image = cv2.blur(gray_image, (2, 2))
    edges = cv2.Canny(grayImage, 100, 160)
    wormMask = None
    if resizeWorm:
        kernel_dilation = np.ones((4, 4), np.uint8)
        img_dilation = cv2.dilate(edges, kernel_dilation, iterations=1)
        kernel_erosion = np.ones((2, 2), np.uint8)
        img_erosion = cv2.erode(img_dilation, kernel_erosion, iterations=1)
        wormMask = img_erosion
    else:
        wormMask = edges
    return wormMask


def random_rotate(img, den, labels, num_examples):

    # takes n patches and creates n*num_examples from each
    result_img = np.zeros(
        [num_examples * len(img), img[0].shape[0], img[0].shape[1], img[0].shape[2]]
    )
    result_den = []
    result_labels = []

    # rotate each patch, num_examples number of times (along with corresponding points)
    for i, patch in enumerate(img):
        for j in range(num_examples):
            patch_mp = [(patch.shape[1]) / 2, (patch.shape[2]) / 2]
            theta = random.randrange(1, 360)
            # create rotation matrix
            sin = torch.sin(torch.Tensor([theta])).item()
            cos = torch.cos(torch.Tensor([theta])).item()

            rot_img = torchvision.transforms.functional.rotate(
                torch.Tensor(patch), theta
            )

            rot_den = []
            points = den[i]
            for p in points:
                xr = (p[0] - patch_mp[0]) * cos - (p[1] - patch_mp[1]) * sin + p[1]
                yr = (p[0] - patch_mp[0]) * sin + (p[1] - patch_mp[1]) * cos + p[0]
                rot_den.append([xr, yr])

            result_img[(i * num_examples) + j] = rot_img
            result_den.append(np.array(rot_den))
            result_labels.append(labels[i])

    return result_img, result_den, result_labels


def equal_crop(img, den, labels, num_patches: int = 4):

    img = img.detach().numpy()
    img_size = img.shape
    # make sure that num_patches is a perfect square
    assert math.isqrt(num_patches) ** 2 == num_patches
    partition = int(math.isqrt(num_patches))
    img_w = img_size[2] // partition
    img_h = img_size[1] // partition
    result_img = np.zeros([num_patches, img_size[0], img_h, img_w])
    result_den = []
    result_lab = []

    start_pos = []
    for x in range(partition):
        for y in range(partition):
            start_pos.append([x * img_h, y * img_w])

    for i, start in enumerate(start_pos):
        start_h = start[0]
        start_w = start[1]
        end_w = start_w + img_w
        end_h = start_h + img_h
        result_img[i] = img[:, start_h:end_h, start_w:end_w]

        # copy the cropped points
        idx = (
            (den[:, 0] >= start_w)
            & (den[:, 0] <= end_w)
            & (den[:, 1] >= start_h)
            & (den[:, 1] <= end_h)
        )
        # shift the coordinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h
        
        record_lab = labels[idx]

        result_den.append(record_den)
        result_lab.append(record_lab)

    return result_img, result_den, result_lab


# random crop augumentation
def random_crop(img, den, labels, edge_image = None, num_patch: int = 4):

    half_h = 512
    half_w = 512
    # half_h = img.size()[1]//4
    # half_w = img.size()[2]//4
    if edge_image == None:
        result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    else: result_img = np.zeros([num_patch*2, img.shape[0], half_h, half_w]) 
    result_den = []
    result_lab = []
    # crop num_patch for each image
    # keep sampling patches until all have non-zero number of samples in them (hence the while loop)
    current_count = 0
    while current_count < num_patch:
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped points
        idx = (
            (den[:, 0] >= start_w)
            & (den[:, 0] <= end_w)
            & (den[:, 1] >= start_h)
            & (den[:, 1] <= end_h)
        )
        # shift the corrdinates
        record_den = den[idx]
        record_lab = labels[idx]
        # gaurentee that each patch sample has point in it 
        if len(record_den) > 0:
            # copy the cropped rect
            result_img[current_count] = img[:, start_h:end_h, start_w:end_w]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
            result_den.append(record_den)
            result_lab.append(record_lab)
            current_count += 1

    return result_img, result_den, result_lab
