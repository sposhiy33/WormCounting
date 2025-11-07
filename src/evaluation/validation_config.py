import os
import sys
import json
import argparse

from src.training.config import get_args_parser

def load_json_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def merge_json_with_args(args, json_config):
    if json_config is None:
        return args
    for key, value in json_config.items():
        setattr(args, key, value)
    return args

def get_arg_parser():
    
    parent_parser = get_args_parser()
    
    parser = argparse.ArgumentParser(parents=[parent_parser])

    # path parameters
    parser.add_argument("--weight_type", type=str, choices=['best_mae.pth', 
                                                            'best_training_loss.pth', 
                                                            'best_mse.pth',
                                                            'best_recall.pth',
                                                            'best_precision.pth',
                                                            'best_f1.pth'])

    parser.add_argument("--result_dir", type=str, help="Path to the result directory of desired validation")
    parser.add_argument("--val_set", type=str, help="Name of the validation set to evaluate on")

    parser.add_argument("--equal_crop", action="store_true", help="Use equal crop during validation")
    parser.add_argument("--median_filter", action="store_true", help="Use median filter during validation")
    return parser

def get_args():
    parser = get_arg_parser()
    args = parser.parse_args()

    json_path = os.path.join(args.result_dir, "args.json")
    json_config = load_json_config(json_path)
    args = merge_json_with_args(args, json_config)
    return args