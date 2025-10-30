import os
import torch
import random
import numpy
import aim
import json

from torch.utils.data import DataLoader, DistributedSampler
from torchinfo import summary

from src.datasets import build_dataset
from src.training.train_utils import *
from src.models import build_model
from src.evaluation.metrics import *
from src.evaluation.vis import *
from src.evaluation.validation import *

class Trainer:
    
    def __init__(self, args):
        
        self.args = args
        self.device = torch.device("cuda")

        ## -- Logger Setup --- ##
        self.aim_run = aim.Run(experiment=args.expname)

        self.aim_run['hparams'] = {
            "backbone": args.backbone,
            "architecture": args.architecture,
            "pointmatch": args.pointmatch,
            "batch_size": args.batch_size,
            "patchsize": args.patch_size,
            "num_patches": args.num_patches,
            "row": args.row,
            "line": args.line,
            "lr": args.lr,
            "lr_backbone": args.lr_backbone,
            "epochs": args.epochs,
            "clip_max_norm": args.clip_max_norm,
            "multiclass": args.multiclass,
            "hsv": args.hsv,
            "hse": args.hse,
            "edges": args.edges,
            "scale": args.scale,
            "sharpness": args.sharpness,
            "equalize": args.equalize,
            "salt_and_pepper": args.salt_and_pepper,
            "loss": args.loss,
            "eos_coef": args.eos_coef,
            "ce_coef": args.ce_coef,
            "matching_label_cost_coef": args.label_loss_coef,
            "matching_point_cost_coef": args.point_loss_coef,
            "dense_loss_coef": args.dense_loss_coef,
            "count_loss_coef": args.count_loss_coef,
            "distance_loss_coef": args.distance_loss_coef,
            "aux_loss_coef": args.aux_loss_coef,
        }

        # list for epoch wise metrics
        self.loss = []
        self.mae = []
        self.mse = []
        self.recall = []
        self.prec = []
        self.f1 = []

        # best metrics -- to be updated throughout training 
        self.min_loss = np.inf
        self.min_mae = np.inf
        self.min_mse = np.inf
        self.max_recall = -np.inf
        self.max_prec = -np.inf
        self.max_f1 = -np.inf

        self.step = 0


        ### --- Directory Setup --- ###


        def make_dir(path: str):
            if os.path.exists(path) == False:
                os.mkdir(path)

        self.result_path = os.path.join(args.output_dir, args.expname)
        make_dir(self.result_path)
        
        self.weight_path = os.path.join(self.result_path, "weights")
        make_dir(self.weight_path)


        # log all args as a json file
        with open(os.path.join(self.result_path, "args.json"), "w") as f:
            json.dump(args.__dict__, f)
            print(f"Args logged to {os.path.join(self.result_path, 'args.json')}")


        ## --- Model Setup --- ##


        self.model, self.criterion = build_model(args, training=True)
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.model_without_ddp = self.model
    
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of params:", n_parameters)
        
        # use different optimation params for different parts of the model
        self.param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.model_without_ddp.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.model_without_ddp.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": args.lr_backbone,
            },
        ]
        
        # Adam is used by default
        self.optimizer = torch.optim.Adam(self.param_dicts, lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=self.args.epochs//5)

        # resume the weights and training state if exists
        if args.resume is not None:
            print("using RESUME")
            checkpoint = torch.load(self.args.resume, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            if (
                not self.args.eval
                and "optimizer" in checkpoint
                and "lr_scheduler" in checkpoint
                and "epoch" in checkpoint
            ):
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                self.args.start_epoch = checkpoint["epoch"] + 1


        ## --- Dataset Setup --- ##


        # create the dataset
        loading_data = build_dataset(args=self.args)
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
            salt_and_pepper=args.salt_and_pepper,
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
        self.data_loader_train = DataLoader(
            train_set,
            batch_sampler=batch_sampler_train,
            collate_fn=utils.collate_fn_crowd,
            num_workers=args.num_workers,
        )

        self.data_loader_val = DataLoader(
            val_set,
            1,
            sampler=sampler_val,
            drop_last=False,
            collate_fn=utils.collate_fn_crowd,
            num_workers=args.num_workers,
        )

        if args.resume is not None:
            print("RESUME CHECKPOINT --> initial eval")
            t1 = time.time()
            result = evaluate_crowd_no_overlap(self.model, self.data_loader_val, self.device)
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


    def train_one_epoch(self, epoch):

        t1 = time.time()
        stat, class_stat = train_one_epoch(
            self.model,
            self.criterion,
            self.data_loader_train,
            self.optimizer,
            self.device,
            self.args.clip_max_norm,
        )

        self.loss.append(stat["loss"])
        # record the training states after every epoch
        t2 = time.time()
        

        ## logging
        self.aim_run.track(stat["loss"], name="loss", step=self.step, context={"subset": "train"})
        self.aim_run.track(self.optimizer.param_groups[0]["lr"], name="lr", step=self.step, context={"subset": "train"})

        # log classwise losses
        for k, v in class_stat.items():
            for i in range(len(v)):
                # get the class name
                if "ce" in k:
                    if i == 0:
                        class_name = "none"
                    else:
                        class_name = self.args.multiclass[i-1]
                if "point" in k:
                    class_name = self.args.multiclass[i]
                
                self.aim_run.track(v[i], name=k + f"_{class_name}", step=self.step, context={"subset": "train"})

        self.save_training_checkpoint(stat)
        
        print(
            "[ep %d][lr %.7f][%.2fs]"
            % (epoch, self.optimizer.param_groups[0]["lr"], t2 - t1)
        )
        # change lr according to the scheduler
        self.lr_scheduler.step()
        

        # run evaluation
        if epoch % self.args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(
                self.model,
                self.data_loader_val,
                self.device,
            )
            t2 = time.time()

            self.save_evaluation_checkpoint(result)

            # update record
            self.mae.append(result[0])
            self.mse.append(result[1])
            self.prec.append(result[2]['4'][0])
            self.recall.append(result[2]['4'][1])
            self.f1.append(result[2]['4'][2])

          
            # logging
            self.aim_run.track(
                {"mae": result[0], "mse": result[1]},
                epoch=self.step,
                context={"subset": "val"}
            )
            self.aim_run.track(
                {
                    "prec": result[2]['4'][0],
                    "recall": result[2]['4'][1],
                    "f1": result[2]['4'][2],
                },
                epoch=self.step,
                context={"subset": "val"}
            )

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
                np.min(self.mae),
            )
            print(
                "=======================================test======================================="
            )
        self.step += 1


    def save_training_checkpoint(self, stat):

        # save latest weights every epoch
        checkpoint_latest_path = os.path.join(self.weight_path, "latest.pth")
        torch.save(
            {
                "model": self.model_without_ddp.state_dict(),
            },
            checkpoint_latest_path,
        )

        # save model with the lowest training loss
        if self.min_loss > stat["loss"]:
            checkpoint_best_path = os.path.join(self.weight_path, "best_training_loss.pth")
            torch.save(
                {
                    "model": self.model_without_ddp.state_dict(),
                },
                checkpoint_best_path,
            )
            # update min loss
            self.min_loss = np.min(self.loss)


    def save_evaluation_checkpoint(self, result):
        
        # save model based on best mae    
        if self.min_mae > result[0]:
            # save model
            checkpoint_best_path = os.path.join(self.weight_path, "best_mae.pth")
            torch.save(
                {
                    "model": self.model_without_ddp.state_dict(),
                },
                checkpoint_best_path,
            )
            # update the new min
            self.min_mae = result[0]

        # save model based on best mse    
        if self.min_mse > result[1]:
            # save model
            checkpoint_best_path = os.path.join(self.weight_path, "best_mse.pth")
            torch.save(
                {
                    "model": self.model_without_ddp.state_dict(),
                },
                checkpoint_best_path,
            )
            # update the new min
            self.min_mse = result[1]


        # save model with best precision performance
        if self.max_prec < result[2]['4'][0]:
            # save model
            checkpoint_best_path = os.path.join(self.weight_path, "best_precision.pth")
            torch.save(
                {
                    "model": self.model_without_ddp.state_dict(),
                },
                checkpoint_best_path,
            )
            # update min
            self.max_prec = result[2]['4'][0] 

        # save model with best recall performance
        if self.max_recall < result[2]['4'][1]:
            # save model
            checkpoint_best_path = os.path.join(self.weight_path, "best_recall.pth")
            torch.save(
                {
                    "model": self.model_without_ddp.state_dict(),
                },
                checkpoint_best_path,
            )
            # update min
            self.max_recall = result[2]['4'][1] 

        # save model with best f1 performance
        if self.max_f1 < result[2]['4'][2]:
            # save model
            checkpoint_best_path = os.path.join(self.weight_path, "best_f1.pth")
            torch.save(
                {
                    "model": self.model_without_ddp.state_dict(),
                },
                checkpoint_best_path,
            )
            # update min
            self.max_f1 = result[2]['4'][2] 