"""
This script does standalone validation using pre-trained model weights.
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

warnings.filterwarnings("ignore")


def get_arg_parser():
    parser = argparse.ArgumentParser("Parameters for testing")
    parser.add_argument("--backbone", type=str, default="vgg16_bn")
    parser.add_argument("--weight_path", type=str)
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--val_set", type=str)

    parser.add_argument(
        "--row", default=2, type=int, help="row number of anchor points"
    )
    parser.add_argument(
        "--line", default=2, type=int, help="line number of anchor points"
    )
    parser.add_argument(
        "--num_classes", type=int, default=1, help="number of non no-person classes"
    )
    parser.add_argument("--dataroot", type=str)
    parser.add_argument("--dataset_file", default="WORM_VAL")
    parser.add_argument(
        "--multiclass",
        action="store_true",
        help="framework to consider using the multiclass framwork or not",
    )
    parser.add_argument(
        "--class_filter",
        type=int,
        default=None,
        help="train on only the specified class index",
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
        "--equal_crop",
        action="store_true",
        help="equally partition the image (for test time)",
    )
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

    # list of datasets to evaluate
    dataset_list = {
        "MULTICLASS": "dataroot/resize_multiclass",
        "L1": "dataroot/resize_L1",
        "ADULT": "dataroot/worm_dataset",
        "MIXED": "dataroot/resize_mixed_eval",
    }
    
    if args.val_set == None:
        for i, key in enumerate(list(dataset_list.keys())):

            print(key)
            dataroot = dataset_list[key]
            # create the training and valiation set
            val_set = loading_data(
                dataroot,
                multiclass=args.multiclass,
                equal_crop=args.equal_crop,
                hsv=args.hsv,
                hse=args.hse,
                edges=args.edges,
                class_filter=args.class_filter,
            )
            # create the sampler used during training
            sampler_val = torch.utils.data.SequentialSampler(val_set)

            result_path = None
            if args.result_dir != None:
                result_path = os.path.join(args.result_dir, f"{key}_vis")
                if os.path.isdir(result_path):
                    shutil.rmtree(result_path)
                os.mkdir(result_path)

            data_loader_val = DataLoader(
                val_set,
                1,
                sampler=sampler_val,
                drop_last=False,
                collate_fn=utils.collate_fn_crowd,
                num_workers=args.num_workers,
            )
            result = evaluate_crowd_no_overlap(
                model,
                data_loader_val,
                device,
                vis_dir=result_path,
                multiclass=args.multiclass,
                num_classes=args.num_classes,
            )

            print(result)
            print("")

    else:
        # create the training and valiation set
        val_set = loading_data(
            args.val_set,
            multiclass=args.multiclass,
            equal_crop=args.equal_crop,
            hsv=args.hsv,
            hse=args.hse,
            edges=args.edges,
            class_filter=args.class_filter,
        )
        # create the sampler used during training
        sampler_val = torch.utils.data.SequentialSampler(val_set)

        result_path = None
        if args.result_dir != None:
            result_path = os.path.join(args.result_dir, f"vis")
            if os.path.isdir(result_path):
                shutil.rmtree(result_path)
            os.mkdir(result_path)

        data_loader_val = DataLoader(
            val_set,
            1,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn_crowd,
            num_workers=args.num_workers,
        )
        result = evaluate_crowd_no_overlap(
            model,
            data_loader_val,
            device,
            vis_dir=result_path,
            multiclass=args.multiclass,
            num_classes=args.num_classes,
        )

        print(result)
        print("")

        
if __name__ == "__main__":
    parser = get_arg_parser()
    arg = parser.parse_args()
    main(arg)
