import argparse
import datetime
import random
import time
from pathlib import Path
import numpy

import torch
from torch.utils.data import DataLoader, DistributedSampler

from crowd_datasets import build_dataset
from engine import *
from models import build_model, build_classifier
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

def get_args_parser():    
    parser = argparse.ArgumentParser('Set parameters for training classifier', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=3500, type=int)
    parser.add_argument('--lr_drop', default=3500, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    parser.add_argument('--pre_weights', type=str)  
    parser.add_argument('--point_weights', type=str,
                        help="Path to pretrained P2PNet for regression")
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--num_classes', default=1, type=int,
                        help="number of non NONE type classes")
    parser.add_argument('--downstream_num_classes', default=2, type=int,
                        help="number of classes for downstream fine grained classification")
    # * Loss coefficients
    parser.add_argument('--point_loss_coef', default=0.0002, type=float)

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")

    parser.add_argument('--set_cost_point', default=0.05, type=float,
                        help="L1 point coefficient in the matching cost")


    # dataset parameters
    parser.add_argument('--dataset_file', default='SHHA')
    parser.add_argument('--data_root', default='./new_public_density_data',
                        help='path where the dataset is')
    
    parser.add_argument('--expname', type=str,
                        help="""folder name under which model weights, tb logs,
                              and any visualiztions will go in the ./results 
                              folder""")
    parser.add_argument('--output_dir', default='./results',
                        help='path where to save, empty for no saving')
    parser.add_argument("--multiclass", default=1, type=int,
                        help="boolean that decides whether class labels should be considered")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

def make_dir(path:str):
    if (os.path.exists(path) == False):
        os.mkdir(path)

def main(args):
    print(args)

    ### Housekeeping
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # specify device
    device = torch.device('cuda')

    ### RESULT SAVING
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    # create folder for result saving
    result_path = os.path.join(args.output_dir, args.expname)
    make_dir(result_path)
    tb_path = os.path.join(result_path, "logs")
    weight_path = os.path.join(result_path, "weights")
    make_dir(tb_path)
    make_dir(weight_path)    
    # make an extra directory meant for visualizations
    vis_path = os.path.join(result_path, "viz")
    make_dir(vis_path)
    # create the logging file    
    run_log_name = os.path.join(result_path, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))
    
    
    ### LOAD P2P Model
    regr_model = build_model(args, training=False)
    # load the trained network
    checkpoint = torch.load(args.point_weights, map_location="cpu")
    regr_model.load_state_dict(checkpoint["model"])
    regr_model.to(device)


    ### BUILD CLASSIFIER 
    model, criterion = build_classifier(args, training=True)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    ### BUILD DATASET
    # create the dataset
    loading_data = build_dataset(args)
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

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
 
    ### TRAINING PARAMS
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    print("Start Training")
    start_time = time.time()

    writer = SummaryWriter(tb_path)
    
    step = 0 
    # training loop
    for epoch in range(args.epochs):
        t1 = time.time()
        stat = train_one_epoch_classifier(regr_model, model, criterion,
                                          data_loader_train, optimizer, device, epoch)
        print(f"Avg Loss:   loss_ce: {stat['loss_ce']}")
        if writer is not None:
            writer.add_scalar('loss/loss_ce', stat['loss_ce'], epoch)
        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))

        lr_scheduler.step()
        checkpoint_latest_path = os.path.join(weight_path, 'latest.pth')
        torch.save({'model': model.state_dict()}, checkpoint_latest_path)

    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser('Point Proposal Classifier training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

