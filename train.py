import argparse
import datetime
import os
import random
import time
import warnings
from pathlib import Path

import numpy
import torch
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torchinfo import summary

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR     '

from crowd_datasets import build_dataset
from engine import *
from models import build_model

warnings.filterwarnings("ignore")

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Set parameters for training P2PNet", add_help=False
    )
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
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

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
        help="Classification weights of each object class, n # of args for n # of classes",
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
        default="./results",
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
        help="option to build model that uses GAT for classification"
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


def make_dir(path: str):
    if os.path.exists(path) == False:
        os.mkdir(path)


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)
    # create folder for result saving
    result_path = os.path.join(args.output_dir, args.expname)
    make_dir(result_path)
    tb_path = os.path.join(result_path, "logs")
    weight_path = os.path.join(result_path, "weights")
    make_dir(tb_path)
    make_dir(weight_path)

    print(args.ce_coef)

    # create the logging file
    run_log_name = os.path.join(result_path, "run_log.txt")
    with open(run_log_name, "w") as log_file:
        log_file.write("Eval Log %s\n" % time.strftime("%c"))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device("cuda")
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # get the P2PNet model
    model, criterion = build_model(args, training=True)
    # send model and criterion to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # resume the weights and training state if exists
    if args.resume is not None:
        print("using RESUME")
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    # create the dataset
    loading_data = build_dataset(args=args)
    # create the training and valiation set
    print(f"Multiclass: {args.multiclass}")
    print(f"HSV: {args.hsv}")
    print(f"HSE: {args.hse}")
    print(f"Edges: {args.edges}")
    train_set, val_set = loading_data(
        args.data_root,
        multiclass=args.multiclass,
        hsv=args.hsv,
        hse=args.hse,
        edges=args.edges,
        scale=args.scale,
        sharpness=args.sharpness,
        equalize=args.equalize,
        patch=True,
        num_patch=args.num_patches,
        patch_size=args.patch_size,
    )

    train_set_stats, val_set_stats = loading_data(
        args.data_root,
        multiclass=args.multiclass,
        hsv=args.hsv,
        hse=args.hse,
        edges=args.edges,
        scale=args.scale,
        sharpness=args.sharpness,
        equalize=args.equalize,
        patch=False,
        num_patch=args.num_patches,
        patch_size=args.patch_size,
    )

    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )
    # the dataloader for training
    data_loader_train = DataLoader(
        train_set,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn_crowd,
        num_workers=args.num_workers,
    )

    data_loader_val = DataLoader(
        val_set,
        1,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn_crowd,
        num_workers=args.num_workers,
    )

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    if args.resume is not None:
        print("RESUME CHECKPOINT --> initial eval")
        t1 = time.time()
        result = evaluate_crowd_no_overlap(model, data_loader_val, device)
        t2 = time.time()

        # print the evaluation results
        print(
            "=======================================test======================================="
        )
        print(
            "mae:",
            result[0],
            "mse:",
            result[1],
            "time:",
            t2 - t1,
        )
        print(
            "=======================================test======================================="
        )

    # print the model summary
    print(summary(model, input_size=(4, 3, 512, 512)))

    print("Start training")
    start_time = time.time()

    # list for epoch wise metrics
    mae = []
    mse = []
    loss = []
    recall = []
    prec = []
    f1 = []

    # best metrics -- to be updated throughout training 
    min_loss = np.inf
    min_mae = np.inf
    min_mse = np.inf
    max_recall = -np.inf
    max_prec = -np.inf
    max_f1 = -np.inf

    # the logger writer
    writer = SummaryWriter(tb_path)

    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        t1 = time.time()
        stat, class_stat = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )

        loss.append(stat["loss"])
        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {}".format(epoch, stat["loss"]))
                if "labels" in args.loss:
                    log_file.write("loss/loss_ce@{}: {}".format(epoch, stat["loss_ce"]))
                if "point" in args.loss:
                    log_file.write(
                        "loss/loss_point@{}: {}".format(epoch, stat["loss_point"])
                    )
                if "density" in args.loss:
                    log_file.write(
                        "loss/loss_dense@{}: {}".format(epoch, stat["loss_dense"])
                    )    
            writer.add_scalar("loss/loss", stat["loss"], epoch)
            if "labels" in args.loss:
                writer.add_scalar("loss/loss_ce", stat["loss_ce"], epoch)
            if "point" in args.loss:
                writer.add_scalar("loss/loss_point", stat["loss_point"], epoch)
            if "density" in args.loss:
                writer.add_scalar("loss/loss_dense", stat["loss_dense"], epoch)
        t2 = time.time()
        print(
            "[ep %d][lr %.7f][%.2fs]"
            % (epoch, optimizer.param_groups[0]["lr"], t2 - t1)
        )
        with open(run_log_name, "a") as log_file:
            log_file.write(
                "[ep %d][lr %.7f][%.2fs]"
                % (epoch, optimizer.param_groups[0]["lr"], t2 - t1)
            )
        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(weight_path, "latest.pth")
        torch.save(
            {
                "model": model_without_ddp.state_dict(),
            },
            checkpoint_latest_path,
        )

        # save model with the lowest training loss
        if min_loss > stat["loss"]:
            checkpoint_best_path = os.path.join(weight_path, "best_training_loss.pth")
            torch.save(
                {
                    "model": model_without_ddp.state_dict(),
                },
                checkpoint_best_path,
            )
            # update min loss
            min_loss = np.min(loss)

        # run evaluation
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(
                model,
                data_loader_val,
                device,
                num_classes=args.num_classes,
                multiclass=args.multiclass,
            )
            t2 = time.time()

            # save model based on best mae    
            if min_mae > result[0]:
                # save model
                checkpoint_best_path = os.path.join(weight_path, "best_mae.pth")
                torch.save(
                    {
                        "model": model_without_ddp.state_dict(),
                    },
                    checkpoint_best_path,
                )
                # update the new min
                min_mae = result[0]

            # save model based on best mse    
            if min_mse > result[1]:
                # save model
                checkpoint_best_path = os.path.join(weight_path, "best_mse.pth")
                torch.save(
                    {
                        "model": model_without_ddp.state_dict(),
                    },
                    checkpoint_best_path,
                )
                # update the new min
                min_mse = result[1]


            # save model with best precision performance
            if max_prec < result[2]['1'][0]:
                # save model
                checkpoint_best_path = os.path.join(weight_path, "best_precision.pth")
                torch.save(
                    {
                        "model": model_without_ddp.state_dict(),
                    },
                    checkpoint_best_path,
                )
                # update min
                max_prec = result[2]['1'][0] 

            # save model with best recall performance
            if max_recall < result[2]['1'][1]:
                # save model
                checkpoint_best_path = os.path.join(weight_path, "best_recall.pth")
                torch.save(
                    {
                        "model": model_without_ddp.state_dict(),
                    },
                    checkpoint_best_path,
                )
                # update min
                max_recall = result[2]['1'][1] 

            # save model with best f1 performance
            if max_f1 < result[2]['1'][2]:
                # save model
                checkpoint_best_path = os.path.join(weight_path, "best_f1.pth")
                torch.save(
                    {
                        "model": model_without_ddp.state_dict(),
                    },
                    checkpoint_best_path,
                )
                # update min
                max_f1 = result[2]['1'][2] 

            mae.append(result[0])
            mse.append(result[1])
            prec.append(result[2]['1'][0])
            recall.append(result[2]['1'][1])
            f1.append(result[2]['1'][2])
            # print the evaluation results
            print(
                "=======================================test======================================="
            )
            print(
                "mae:",
                result[0],
                "mse:",
                result[1],
                "prec/rec/f1",
                result[2],
                "time:",
                t2 - t1,
                "best mae:",
                np.min(mae),
            )
            with open(run_log_name, "a") as log_file:
                log_file.write(
                    "mae:{}, mse:{}, time:{}, best mae:{}".format(
                        result[0], result[1], t2 - t1, np.min(mae)
                    )
                )
            print(
                "=======================================test======================================="
            )
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {}".format(step, result[0]))
                    log_file.write("metric/mse@{}: {}".format(step, result[1]))
                writer.add_scalar("metric/mae", result[0], step)
                writer.add_scalar("metric/mse", result[1], step)
                step += 1

    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def avg_class_loss(loss, writer, epoch):
    ce_losses = []
    point_losses = []
    for entry in loss:
        # calculate avg ce loss
        ce_list = entry["class_loss_ce"]
        ce_list = [i.item() for i in ce_list]
        ce_losses.append(ce_list)

        # calculate point losses
        point_list = entry["class_loss_point"]
        point_list = [i.item() for i in point_list]
        point_losses.append(point_list)

    ce_losses, point_losses = numpy.array(ce_losses), numpy.array(point_losses)
    ce_losses, point_losses = ce_losses.transpose(), point_losses.transpose()
    ce_mean = numpy.nanmean(ce_losses, axis=1)
    point_mean = numpy.nanmean(point_losses, axis=1)
    ce_mean, point_mean = ce_mean.tolist(), point_mean.tolist()

    for i, mean in enumerate(ce_mean):
        writer.add_scalar(f"metric/class{i}_loss_ce", mean, epoch)
    for i, mean in enumerate(point_mean):
        writer.add_scalar(f"metric/class{i}_loss_point", mean, epoch)

    return ce_mean, point_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "P2PNet training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)