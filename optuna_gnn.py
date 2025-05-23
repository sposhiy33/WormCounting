"Hyperparameter optimization with Optuna"

import optuna
from optuna.trial import TrialState

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

from crowd_datasets import build_dataset
from engine import *
from models import build_model

from train import get_args_parser, make_dir

# global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_optim(trial, args):

    # Hyperparameters to optimize
    gnn_layers = trial.suggest_int("gnn_layers", 1, 5)
    neighbors = trial.suggest_int("neighbors", 1, 15)

    args.gnn_layers = gnn_layers
    args.knn = neighbors

    model, criterion = build_model(args, training=True)

    return model, criterion

def make_dir(path: str):
    if os.path.exists(path) == False:
        os.mkdir(path)

# truncated main loop from train.py
def objective(trial):

    # create folder for result saving
    result_path = os.path.join(args.output_dir, args.expname)
    make_dir(result_path)
    tb_path = os.path.join(result_path, "logs")
    weight_path = os.path.join(result_path, "weights")
    make_dir(tb_path)
    make_dir(weight_path)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # get the P2PNet model
    model, criterion = model_optim(trial, args)
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

    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # create the dataset
    loading_data = build_dataset(args=args)
    
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

    print("Start training")
    # save the performance during the training
    mae = []
    mse = []
    loss = []
    min_loss = 100.0

    step = 0
    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        
        # forward and backward pass
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
        t2 = time.time()
        print(
            "[ep %d][lr %.7f][%.2fs]"
            % (epoch, optimizer.param_groups[0]["lr"], t2 - t1)
        )
        
        # change lr according to the scheduler
        lr_scheduler.step()

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

            mae.append(result[0])
            mse.append(result[1])
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
                "best mae:",
                np.min(mae),
            )
            
            print(
                "=======================================test======================================="
            )


            # save the best model with best average count error
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(weight_path, "best_mae.pth")
                torch.save(
                    {
                        "model": model_without_ddp.state_dict(),
                    },
                    checkpoint_best_path,
                )

    trial.report(np.min(mae), np.argmin(mae))
    
    # Handle pruning based on the intermediate value.
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()

    return np.min(mae)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Optuna hyperparameter optimization", parents=[get_args_parser()])

    # set arguments as a global variable 
    global args
    args = parser.parse_args()

    # create the study
    study = optuna.create_study(study_name="GAT", storage="sqlite:///gat.db", load_if_exists=True, direction="minimize")
    # optimize the objective function
    study.optimize(objective, n_trials=20, gc_after_trial=True)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

