"""
This script does standalone validation using pre-trained model weights.
"""
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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
import torchvision.transforms as standard_transforms

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'

from src.datasets import build_dataset
from src.models import build_model
from src.models.matcher import build_matcher_crowd
from src.evaluation.metrics import evaluate_crowd_no_overlap
from src.evaluation.validation_config import get_args

import src.util.misc as utils


def main(args):

    device = 'cuda'

    # argument parsing
    model = build_model(args)
    matcher = build_matcher_crowd(args)
    weight_path = os.path.join(args.result_dir,f"weights/{args.weight_type}")


    # load the trained network
    checkpoint = torch.load(weight_path, map_location="cpu")
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

    # list of datasets to evaluate
    dataset_list = {
        # "MULTICLASS": "dataroot/resize_multiclass",
        #"L1": "dataroot/resize_L1",
        "LOWRES IMAGES": "dataroot/lowres_all_images",
        # "HIGH_RES": "dataroot/all_images",
    }


    # if not specific evaluation set is specified, evaluated on 
    # the sets given in 'datset_list'
    if args.val_set == None:
        for key in list(dataset_list.keys()):

            print(key)
            dataroot = dataset_list[key]
            # create the training and valiation set
            train_set, val_set = loading_data(
                dataroot,
                multiclass=args.multiclass,
                equal_crop=args.equal_crop,
                hsv=args.hsv,
                hse=args.hse,
                edges=args.edges,
                equalize=args.equalize,
                num_patch=args.num_patches,
                patch_size=args.patch_size,
                median_filter=args.median_filter,
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


            # if specified, evalutate on the training images,
            # this is mainly for sanity checks and checking for overfitting
            if args.eval_train:
                print("TRAIN")
      
                # create the sampler used during training
                sampler_train = torch.utils.data.SequentialSampler(train_set)

                result_path = None
                if args.result_dir != None:
                    result_path = os.path.join(args.result_dir, f"TRAIN_vis")
                    if os.path.isdir(result_path):
                        shutil.rmtree(result_path)
                    os.mkdir(result_path)

                data_loader_train = DataLoader(
                    train_set,
                    1,
                    sampler=sampler_train,
                    drop_last=False,
                    collate_fn=utils.collate_fn_crowd,
                    num_workers=args.num_workers,
                )
                result = evaluate_crowd_no_overlap(
                    model,
                    data_loader_train,
                    device,
                    vis_dir=result_path,
                )

                print(result)
                print("")

    # otherwise, use dataset specified in args.val_set
    else:
        # create the training and valiation set
        train_set, val_set = loading_data(
            args.val_set,
            multiclass=args.multiclass,
            equal_crop=args.equal_crop,
            hsv=args.hsv,
            hse=args.hse,
            edges=args.edges,
            equalize=args.equalize,
            num_patch=args.num_patches,
            patch_size=args.patch_size,
            median_filter=args.median_filter,
        )
        # create the sampler used during training
        sampler_val = torch.utils.data.SequentialSampler(val_set)

        weight_metric = args.weight_type.replace("best_", "").replace(".pth", "")
        print(weight_metric)

        result_path = None
        if args.result_dir != None:
            result_path = os.path.join(args.result_dir, f"{weight_metric}_vis")
            if os.path.isdir(result_path):
                shutil.rmtree(result_path)
            os.mkdir(result_path)

        print(result_path)

        data_loader_val = DataLoader(
            val_set,
            1,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn_crowd,
            num_workers=args.num_workers,
        )

        print(f"Validation set size: {len(val_set)}")
        if len(val_set) == 0:
            print("ERROR: Validation set is empty! Check dataset path and file lists.")
            return

        result = evaluate_crowd_no_overlap(
            model,
            data_loader_val,
            device,
            vis_dir=result_path,
        )

        print(result)
        print("")

        
if __name__ == "__main__":
    args = get_args()
    main(args)