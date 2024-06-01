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
                            scale=False, rotate=False, patch=False, flip=False):
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
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, point, label_class = load_data((img_path, gt_path), self.train)
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
            img, point = random_crop(img, point)

            # convert point arrays for each image to torch Tensor type
            for i, _ in enumerate(point):
                point[i] = torch.Tensor(point[i])

            # apply rotation transformation for data balacing
            if self.rotate: 
                patch_expansion = 2
                if label_class == "L1": patch_expansion=1
                elif label_class == "ADT": patch_expansion=8
                img, point = random_rotate(img, point, patch_expansion)
        
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            for i, _ in enumerate(point):
                point[i][:, 0] = 128 - point[i][:, 0]

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
           
            # for multiclass classification, assign labels 
            if label_class is not None:
                labels = torch.zeros([point[i].shape[0]]).long() 
                for l in range(labels.size()[0]): 
                    if label_class == "L1":
                        labels[l] = 1
                    elif label_class == "ADT":
                        labels[l] = 2
            else:
                labels = torch.ones([point[i].shape[0]]).long()       

            target[i]['labels'] = labels
        
        return img, target


def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load ground truth points
    points = []  

    # assign a class
    label_class = None
    if "L1" in img_path:
        label_class = "L1"
    elif "ADT" in img_path:
        label_class = "ADT"

    with open(gt_path) as f_label:
        for line in f_label:
            if "\t" in line:
                x = float(line.strip().split("\t")[0].strip()) 
                y = float(line.strip().split("\t")[1].strip())
                points.append([x,y])
            else:
                x = float(line.strip().split(' ')[0].replace(",", ""))
                y = float(line.strip().split(' ')[1])
                points.append([x, y])
        
    return img, np.array(points), label_class

def random_rotate(img, den, num_examples):
    
    # takes n patches and creates n*num_examples from each
    result_img = np.zeros([num_examples*len(img), img[0].shape[0], img[0].shape[1], img[0].shape[2]])
    result_den = []

    # rotate each patch, num_examples number of times (along with corresponding points)
    for i,patch in enumerate(img):
        for j in range(num_examples): 
            ang = random.randrange(1,360)
            rot_img = torchvision.transforms.functional.rotate(torch.Tensor(patch), ang)
            rot_den = torchvision.transforms.functional.rotate(torch.unsqueeze(den[i],0), ang)
            
            result_img[(i*num_examples) + j] = rot_img
            result_den.append(torch.squeeze(rot_den, 0))

    return result_img, result_den
             
# random crop augumentation
def random_crop(img, den, num_patch=4):

    half_h = img.size()[1]//4
    half_w = img.size()[2]//4
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = []
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

        if len(record_den) > 0:
            # copy the cropped rect
            result_img[current_count] = img[:, start_h:end_h, start_w:end_w]
            record_den[:, 0] -= start_w
            record_den[:, 1] -= start_h
            result_den.append(record_den)
            current_count += 1 

    return result_img, result_den
