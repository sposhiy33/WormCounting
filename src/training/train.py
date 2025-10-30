import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# suppress opencv warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR     '
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

import argparse
import datetime
import random
import time
import numpy as np
import torch

from src.training.trainer import Trainer
from src.training.config import get_args
import src.util.misc as utils


def main(args):

    # environment setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.gpu_id)
    print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device("cuda")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("Start training")

    # create the trainer
    trainer = Trainer(args)

    start_time = time.time()

    # training starts here
    for epoch in range(args.start_epoch, args.epochs):
        trainer.train_one_epoch(epoch)

    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    # load arguments
    args = get_args()

    main(args)