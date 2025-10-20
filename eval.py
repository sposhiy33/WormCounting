"""
Pure evaluation script for unlabelled images
"""

import argparse
import datetime
import os
import random
import shutil
import time
import warnings
from pathlib import Path

import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model

def get_arg_parser():
    parser = argparse.ArgumentParser("Parameters for eval")
    parser.add_argument("--backbone", type=str, default="vgg16_bn")

    # path parameters
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--weight_type", type=str, choices=['best_mae.pth', 'best_training_loss.pth', "best_f1.pth"])
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--val_set", type=str)

    # model parameters
    parser.add_argument(
        "--row", default=2, type=int, help="row number of anchor points"
    )
    parser.add_argument(
        "--line", default=2, type=int, help="line number of anchor points"
    )
    parser.add_argument(
        "--num_classes", type=int, default=1, help="number of non no-person classes"
    )

    # dataset parametersS
    parser.add_argument("--dataroot", type=str)
    parser.add_argument("--dataset_file", default="WORM_EVAL")
    
    parser.add_argument(
        "--multiclass",
        nargs="+",
        type=str,
        help="name of the classes",
    )

    parser.add_argument(
        "--hsv",
        action="store_true",
        help="use Hue, Saturation, Value respectively for channels, by default RGB is used",
    )
    parser.add_argument(
        "--hse",
        action="store_true",
        help="use Hue, Saturation, Edges respectively for channels, by default RGB is used",
    )
    parser.add_argument(
        "--edges", action="store_true", help="use edge detection photos"
    )

    parser.add_argument(
        "--eval_train", action="store_true", help="validate on the training images. For debugging purposes"
    )
    parser.add_argument(
        "--equal_crop",
        action="store_true",
        help="equally partition the image (for test time)",
    )

    # architecture parameterss
    parser.add_argument("--noreg",
        action="store_true",
        help="set regression branch to zero, so that ground truth points are not offset,\
                used for debugging pruposes")
    
    parser.add_argument(
        "--classifier",
        action="store_true",
        help="option to intialize network with only the classifiation branch"
    )

    parser.add_argument(
        "--mlp",
        action='store_true',
        help="option to build a model with MLP for classification and point offest prediction"
    )

    parser.add_argument(
        "--mlp_classifier",
        action='store_true',
        help="option to build a model with MLP for classification only predicition"
    )

    parser.add_argument(
        "--gat",
        action='store_true',
        help="option to build a model with GAT"
    )



    ## extra args
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    return parser

def main(args):
    device = torch.device("cuda")

    # argument parsing
    model = build_model(args)
    args.num_classes = 1
    args.weight_path = os.path.join(args.result_dir,f"weights/{args.weight_type}")


    # load the trained network
    checkpoint = torch.load(args.weight_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    model.to(device)
    model.eval()

    # create the pre-processing transform
    transform = standard_transforms.Compose(
        [
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    ### BUILD DATASET ###
    # create the dataset
    loading_data = build_dataset(args)
    print(args)

    # create the training and valiation set
    val_set = loading_data(
        args.val_set,
    )
    # create the sampler used during training
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    result_path = None
    if args.result_dir != None:
        result_path = os.path.join(args.result_dir, f"viz")
        if os.path.isdir(result_path):
            shutil.rmtree(result_path)
        os.mkdir(result_path)

    data_loader_val = DataLoader(
        val_set,
        1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
 
    vis_dir = args.val_set.split("/")[-1]
    if vis_dir == '': vis_dir = args.val_set.split("/")[-2]
    vis_dir = os.path.join("eval_viz", f"{vis_dir}_viz")

    if os.path.isdir(vis_dir):
        shutil.rmtree(vis_dir)
    os.mkdir(vis_dir) 

    result = evaluate_crowd(
        model,
        data_loader_val,
        device,
        vis_dir = vis_dir,
    )

    print(result)
    print("")

if __name__ == "__main__":
    parser = get_arg_parser()
    arg = parser.parse_args()
    main(arg)


