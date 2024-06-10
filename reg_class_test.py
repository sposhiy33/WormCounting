"""
Evaluation of standalone classification model
"""

import argparse
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model, build_classifier
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_arg_parser():
    parser = argparse.ArgumentParser("Parameters for testing")
    parser.add_argument("--backbone", type=str, default="vgg16_bn")
    parser.add_argument("--point_weights", type=str,
                        help="Path of model checkpoint of regression model")
    parser.add_argument("--class_weights", type=str,
                        help="Path to model checkpoint of fine grained classification model")
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--multiclass", type=bool, default=True,
                        help="whether or not class labels are ingorned upon intializing the dataset")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument("--num_classes", type=int, default=1,
                        help="number of non no-person classes")
    parser.add_argument("--downstream_num_classes", type=int, default=2,
                        help="number of classes for fine grained classification")
    parser.add_argument("--dataroot", type=str)
    parser.add_argument("--dataset_file", default="SHHA")

    ## throwaway args
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    return parser

def main(args):
    device = torch.device("cuda")
    
    regr_model = build_model(args, training=False)
    class_model = build_classifier(args, training=False)
        
    # load the trained network
    regr_checkpoint = torch.load(args.point_weights, map_location="cpu")
    regr_model.load_state_dict(regr_checkpoint["model"])
    class_checkpoint = torch.load(args.class_weights, map_location="cpu")
    class_model.load_state_dict(class_checkpoint["model"])

    regr_model.to(device)
    class_model.to(device)
    regr_model.eval()
    class_model.eval()

    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ### BUILD DATASET ###
    # create the dataset
    loading_data = build_dataset(args)
    
    print(args)

    # create the training and valiation set
    train_set, val_set = loading_data(args.dataroot, args.multiclass)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, 1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    result = evaluate_crowd_w_fine_grained(regr_model=regr_model, 
                                           class_model=class_model,
                                           data_loader=data_loader_val, 
                                           device=device, vis_dir=args.result_dir)
    
    print(result) 


if __name__ == "__main__":
    parser = get_arg_parser()
    arg = parser.parse_args()
    main(arg)

