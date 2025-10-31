import os
import argparse
import yaml


def load_yaml_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def merge_yaml_with_args(args, yaml_config):
    """Merge YAML configuration with command line arguments."""

    if yaml_config is None:
        return args

    # flatten any nested dictionaries in the YAML configuration
    flat_config = {}
    for key, value in yaml_config.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                flat_config[subkey] = subvalue
        else:
            flat_config[key] = value

    # Iterate through the YAML configuration and update the command line arguments
    for key, value in flat_config.items():
        setattr(args, key, value)

    return args

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Set parameters for training P2PNet", add_help=False
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to YAML configuration file'
    )

    # * Experiment configuration
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--lr_backbone", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=3500, type=int)
    parser.add_argument("--lr_drop", default=3500, type=int)
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )

    # Model parameters
    parser.add_argument(
        "--pre_weights",
        type=str,
        default=None,
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="vgg16_bn",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--num_classes", default=1, type=int, help="number of non NONE type classes"
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=1,
        type=float,
        help="Class coefficient in the matching cost",
    )

    parser.add_argument(
        "--set_cost_point",
        default=0.05,
        type=float,
        help="L1 point coefficient in the matching cot",
    )

    parser.add_argument(
        "--loss",
        nargs="+",
        type=str,
        help="specify which terms to include in the loss",
        default=["labels", "points"],
        choices=["labels", "points", "density", "count", "distance", "aux"],
    )
    
    # * Loss coefficients (guide training scheme)
    
    parser.add_argument(
        "--label_loss_coef",
        default=1,
        type=float,
        help="loss weight of CE loss"
    )
    
    parser.add_argument(
        "--point_loss_coef", 
        default=0.0002, 
        type=float,
        help="loss weight of regression loss"
    )

    parser.add_argument(
        "--dense_loss_coef",
        default=1,
        type=float,
        help="loss weight of dense estimation loss",
    )
    parser.add_argument(
        "--count_loss_coef",
        default=1,
        type=float,
        help="loss weight of count estimation loss",
    )
    
    parser.add_argument(
        "--distance_loss_coef",
        default=1,
        type=float,
        help="loss weight of distance regulation term",
    )

    parser.add_argument(
        "--aux_loss_coef",
        default=1,
        type=float,
        help="loss weight of auxillary points guidance term",
    )

    parser.add_argument(
        "--eos_coef",
        default=0.5,
        type=float,
        help="Relative classification weight of the no-object class",
    )

    parser.add_argument(
        "--ce_coef",
        nargs="+",
        type=float,
        help="Classification weights of each object class, n # of args for n # of classes, provide list object",
    )

    parser.add_argument(
        "--map_res",
        default=4,
        type=int,
        help="resoltion down sampling factor (each axis), total downsample will be map_res^2",
    )
    parser.add_argument(
        "--gauss_kernel_res",
        default=9,
        type=int,
        help="kernel size for generating heatmaps",
    )

    # --- debris loss parameters ---
    parser.add_argument(
        "--debris_class_idx",
        default=None,
        type=int,
        help="class index of the debris class",
    )
    parser.add_argument(
        "--debris_radius",
        default=5,
        type=float,
        help="radius of the debris class",
    )
    parser.add_argument(
        "--neg_lambda_debris",
        default=1,
        type=float,
        help="weight of the debris class",
    )
    parser.add_argument(
        "--neg_lambda_other",
        default=1,
        type=float,
        help="weight of the general backgroud class",
    )

    parser.add_argument(
        "--row", default=3, type=int, help="row number of anchor points"
    )
    
    parser.add_argument(
        "--line", default=3, type=int, help="line number of anchor points"
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="WORM")
    parser.add_argument(
        "--data_root",
        default="./new_public_density_data",
        help="path where the dataset is",
    )

    parser.add_argument(
        "--expname",
        type=str,
        help="""folder name under which model weights, tb logs,
                              and any visualiztions will go in the ./results 
                              folder""",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/results",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--multiclass",
        nargs="+",
        type=str,
        help="name of the classes",
    )

    parser.add_argument(
        "--hsv",
        action="store_true",
        help="use HSV channels during training",
    )
    parser.add_argument(
        "--hse", action="store_true", help="use HSE channels during training"
    )
    parser.add_argument(
        "--edges", action="store_true", help="use Canny edge output during training"
    )
    parser.add_argument(
        "--sharpness", action="store_true", help="use sharpness data augmnetation during training"
    )
    parser.add_argument(
        "--scale", action="store_true", help="use scale data augmentation during training"
    )
    parser.add_argument(
        "--equalize", action="store_true", help="use histogram equalization during training"
    )
    parser.add_argument(
        "--salt_and_pepper", action="store_true", help="use salt and pepper noise during training"
    )
    parser.add_argument(
        "--num_patches", type=int, default=4, help="number of patches to samples from each image"
    )
    parser.add_argument(
        "--patch_size", type=int, default=512, help="size of the patches"
    )

    parser.add_argument("--gnn_layers", default=1, type=int, help="number of GNN layers")

    parser.add_argument("--knn", default=4, type=int, help="number of nearest neighbors for KNN graph construction")

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from checkpoint"
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )

    parser.add_argument(
        "--pointmatch",
        action="store_true",
        help="only use point distance as the hungarian alg metric"
    )

    parser.add_argument(
        "--noreg",
        action="store_true",
        help="set regression branch to zero, so that ground truth points are not offset,\
                used for debugging pruposes"
    )

    parser.add_argument(
        "--architecture",
        type=str,
        help="option to build a model with specific architecture",
        choices=["p2p", "classifier", "mlp", "mlp_classifier", "gat"],
        default="p2p",
    )

    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    
    parser.add_argument(
        "--eval_freq",
        default=5,
        type=int,
        help="frequency of evaluation, default setting is evaluating in every 5 epoch",
    )
    parser.add_argument(
        "--gpu_id", default=0, type=int, help="the gpu used for training"
    )

    return parser

def get_args():
    """Main function to get parsed and merged arguments."""
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Load YAML config if provided
    yaml_config = None
    if args.config:
        yaml_config = load_yaml_config(args.config)
        args = merge_yaml_with_args(args, yaml_config)

    # internal validation
    args.num_classes = len(args.multiclass)
    args.debris_class_idx = args.multiclass.index("debris") + 1 if "debris" in args.multiclass else None
    
    # if debris is a class, make sure it is a not an output class in the logit
    if args.debris_class_idx is not None:
        args.num_classes -= 1
    
    return args



