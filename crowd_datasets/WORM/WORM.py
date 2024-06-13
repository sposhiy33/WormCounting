import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from PIL import Image
import cv2
import glob
import scipy.io as io

class WORM(Dataset):
    def __init__(self, data_root, transform=None, train=False, 
                            scale=False, rotate=False, patch=False, flip=False, multiclass=False, hsv=False):
        self.root_path = data_root
        
        if "worm_dataset" in self.root_path:
            self.train_lists = "shtrain.list"
            self.eval_list = "shtest.list"
        else:
            self.train_lists = "mtrain.txt"
            self.eval_list = "mtest.txt"

        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        #loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2: 
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                                    os.path.join(self.root_path, line[1].strip())
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
        self.hsv = hsv
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point, labels = load_data((img_path, gt_path), self.train, self.multiclass)
        # apply augumentation
        if self.transform is not None:
            img = self.transform(img)

        if self.train and self.scale:
            # data augmentation -> random scale
            scale_range = [0.7, 1.3]
            min_size = min(img.shape[1:])
            scale = random.uniform(*scale_range)
            # scale the image and points
            if scale * min_size > 128:
                img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=scale).squeeze(0)
                point *= scale
        
        # random crop augumentaiton
        if self.train and self.patch:
            img, point, labels = random_crop(img, point, labels)

            # convert point arrays for each image to torch Tensor type
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

        
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]
        
        if self.hsv:
            img = rgb_to_hsv(img)

        if not self.train:
            point = [point]

        img = torch.Tensor(img)
        # pack up related infos
        target = [{} for i in range(len(point))]
        for i, _ in enumerate(point):
            target[i]['point'] = torch.Tensor(point[i])
            image_id = int(img_path.split('/')[-1].split('.')[0].split('_')[-1])
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.Tensor(labels[i].tolist()).long()
        
        return img, target


def load_data(img_gt_path, train, multiclass):
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
                points.append([x,y])
                
                # create labels
                if multiclass:
                    # if the label is included in the point txt file, use this scheme
                    if elements == 3:
                        lab = str(line.strip().split("\t")[2].strip())
                        if lab == "Gravid": labels.append(2)
                        elif lab == "L1": labels.append(1)
                    # else infer label from the img file name
                    else:
                        if "L1" in img_path: labels.append(1)
                        elif "ADT" in img_path: labels.append(2)
                else: labels.append(1)


            else:
                x = float(line.strip().split(' ')[0].replace(",", ""))
                y = float(line.strip().split(' ')[1])
                points.append([x, y])
    return img, np.array(points), np.array(labels)

def rgb_to_hsv(rgb):
    """
    Implementation taken from: https://github.com/limacv/RGB_HSV_HSL/blob/master/color_torch.py
    Parameters:
    img --> torch.Tensor of shape 
    """
    print("HSV")
    rgb = torch.Tensor(rgb)
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    return torch.cat([hsv_h, hsv_s, hsv_v], dim=1)


def random_rotate(img, den, num_examples):
    
    # takes n patches and creates n*num_examples from each
    result_img = np.zeros([num_examples*len(img), img[0].shape[0], img[0].shape[1], img[0].shape[2]])
    result_den = []

    # rotate each patch, num_examples number of times (along with corresponding points)
    for i,patch in enumerate(img):
        for j in range(num_examples): 
            theta = random.randrange(1,360)
            # create rotation matrix
            sin = torch.sin(torch.Tensor([theta])).item()
            cos = torch.cos(torch.Tensor([theta])).item()
            rot_matrix = torch.Tensor([[cos, -sin],[sin, cos]])
            
            rot_img = torchvision.transforms.functional.rotate(torch.Tensor(patch), theta)
            rot_den = torch.matmul(den[i], rot_matrix)
            print(rot_den.size()) 
            result_img[(i*num_examples) + j] = rot_img
            result_den.append(rot_den)

    return result_img, result_den
             
# random crop augumentation
def random_crop(img, den, labels, num_patch=4):

    half_h = img.size()[1]//4
    half_w = img.size()[2]//4
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
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
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_lab = labels[idx]

        if len(record_den) > 0:
            # copy the cropped rect
            result_img[current_count] = img[:, start_h:end_h, start_w:end_w]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
            result_den.append(record_den)
            result_lab.append(record_lab)
            current_count += 1 

    return result_img, result_den, result_lab
