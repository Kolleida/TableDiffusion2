from pathlib import Path
import os
from __future__ import print_function
import argparse
import os
from datetime import datetime
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mlflow
import sys
import time
import argparse

from tablediffusion.models import *
import tablediffusion.utilities.utils as utils
import tablediffusion.utilities.data_utils as data_utils
import tablediffusion.config.configs as configs
import tablediffusion.config as config
from tablediffusion.utilities import run_synthesisers



def do_things(input_dataset, output_dir="/home/azureuser/drive1/syn"):
    SRCDIR = "tablediffusion"
    DIR = Path("stuff")
    DATADIR = Path("/home/azureuser/drive1/data/")
    RESULTDIR = DIR / "results"

    sys.path.append(str(SRCDIR))

    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    SEED = 999

    for p in [SRCDIR, DIR, DATADIR, RESULTDIR]:
        if not os.path.exists(p):
            print(f"{p} does not exist")

    EPOCHS = 1
    DIFFUSION_STEPS = 3

    synthesisers = {
        "DPWGAN_Synthesiser": (
            WGAN_Synthesiser,
            {
                "batch_size": 1024,
                'gen_lr': 0.005,
                'dis_lr': 0.001,
                "latent_dim": 128,
                'n_critic': 2,
                "epoch_target": EPOCHS,
                "mlflow_logging": False,
                'gen_dims': (512, 512),
                'dis_dims': (512, 512)
            },
            {
                "n_epochs": EPOCHS,
            },
            {
                "use_raw_data": False,
            },
        ),
        "TableDiffusion_Synthesiser": (
            TableDiffusion_Synthesiser,
            {
                "batch_size": 1024,
                "lr": 0.005,
                "dims": (512, 512),
                "mlflow_logging": False,
                "epoch_target": EPOCHS * DIFFUSION_STEPS,
                "diffusion_steps": DIFFUSION_STEPS,
                "predict_noise": True,
            },
            {
                "n_epochs": EPOCHS,
                "verbose": True,
            },
            {
                "use_raw_data": True,
            },
        )
    }

    dset_name = input_dataset
    datasets = {dset_name: config.datasets[dset_name]}

    exp_hash = datetime.now().strftime("%y%m%d_%H%M%S")
    EXP_NAME = f"exp_{exp_hash}"

    # Make directories for experiment EXP_NAME
    EXP_PATH = RESULTDIR / EXP_NAME
    # FAKE_DSET_PATH = EXP_PATH / "fake_datasets"
    FAKE_DSET_PATH = Path(output_dir)
    if not os.path.exists(FAKE_DSET_PATH):
        os.makedirs(FAKE_DSET_PATH)

    exp_id = mlflow.create_experiment(f"{EXP_NAME}")

    print(f"\n\nRunning experiment: {EXP_NAME}\n\n")

    start = time.time()

    run_synthesisers(
        datasets=datasets,
        synthesisers=synthesisers,
        exp_name=EXP_NAME,
        exp_id=exp_id,
        datadir=DATADIR,
        repodir="./",
        epsilon_values=[2.0],
        repeats=1,
        metaseed=SEED,
        generate_fakes=True,
        fake_sample_path=EXP_PATH / "samples",
        fake_data_path=FAKE_DSET_PATH,
        cuda=True,
    )

    mlflow.end_run()

    end = time.time()

    print(f'Time Elapsed: {(end - start) / 60} min')


if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Run DP baselines for WGAN and Diffusion.')
    parser.add_argument('--input_dataset', required=True, help='Name of dataset. Must be defined in configs.py')
    parser.add_argument('--output_dir', default="/home/azureuser/drive1/syn", help='Directory to put generated synthetic data.')
    args = parser.parse_args()

    do_things(args.input_dataset, args.output_dir)