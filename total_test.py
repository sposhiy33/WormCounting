import argparse
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_arg_parser():
    parser = argparse.ArgumentParser("Parameters for testing")
    parser.add_argument("--backbone", type=str, default="vgg16_bn")
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--result_dir", type=str)

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument("--data_root", type=str)
    parser.add_argument("--dataset_file", default="SHHA")

    ## throwaway args
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    return parser

def main(args):

    device = torch.device("cuda")
    model = build_model(args)
   
    # load the trained network
    checkpoint = torch.load(args.weight_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    
    model.to(device)
    model.eval()

    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    ### BUILD DATASET ###
    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    train_set, val_set = loading_data(args.data_root)
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

    result = evaluate_crowd_no_overlap(model, data_loader_val, device)
    
    print(result) 


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)

