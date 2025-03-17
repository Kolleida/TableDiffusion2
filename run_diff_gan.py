from pathlib import Path
import os
import argparse
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

# Have to add path to the top level module, because the code doesn't seem to use relative imports...
REPO_DIR = os.path.dirname(os.path.realpath(__file__))
SRCDIR = os.path.join(REPO_DIR, 'tablediffusion')
sys.path.append(SRCDIR)

# TableDiffusion code.
from tablediffusion.models import *
import tablediffusion.utilities.utils as utils
import tablediffusion.utilities.data_utils as data_utils
import tablediffusion.config.configs as configs
import tablediffusion.config as config
from tablediffusion.utilities import run_synthesisers


GAN_EPOCHS = 1
DIFF_EPOCHS = 1
DIFFUSION_STEPS = 3

SYNTHESIZERS = {
    "DPWGAN": (
        WGAN_Synthesiser,
        {
            "batch_size": 1024,
            'gen_lr': 0.005,
            'dis_lr': 0.001,
            "latent_dim": 128,
            'n_critic': 2,
            "epoch_target": GAN_EPOCHS,
            "mlflow_logging": False,
            'gen_dims': (512, 512),
            'dis_dims': (512, 512)
        },
        {
            "n_epochs": GAN_EPOCHS,
        },
        {
            "use_raw_data": False,
        },
    ),
    "DPautoGAN": (
        DPautoGAN_Synthesiser,
        {
            'batch_size': 1024,
            'latent_dim': 64,
            "gen_dims": (256, 256, 256),
            "dis_dims": (256, 256, 256),
            'gen_lr': 0.0001,
            'dis_lr': 0.0007,
            "ae_lr": 0.02,
            "ae_compress_dim": 16,
            "ae_eps_frac": 0.3,
            'epoch_target': GAN_EPOCHS,
            'mlflow_logging': True,
        },
        {
            "n_epochs": GAN_EPOCHS,
        },
        {
            "use_raw_data": False,
        },
    ),
    "PATEGAN": (
        PATEGAN_Synthesiser,
        {
            'batch_size': 1024,
            "gen_dims": (512, 512, 512),
            "dis_dims": (512, 512, 512),
            'gen_lr': 0.005,
            'dis_lr': 0.001,
            'latent_dim': 128,
            'num_teachers': 30,
            'teacher_iters': 8,
            'student_iters': 5,
            'epoch_target': GAN_EPOCHS,
            'mlflow_logging': True,
        },
        {
            'n_epochs': GAN_EPOCHS,
            'noise_multiplier': 0.0048,
        },
        {
            "use_raw_data": False,
        },
    ),
    "DPDiffusion": (
        TableDiffusion_Synthesiser,
        {
            "batch_size": 1024,
            "lr": 0.005,
            "dims": (512, 512),
            "mlflow_logging": False,
            "epoch_target": DIFF_EPOCHS,
            "diffusion_steps": DIFFUSION_STEPS,
            "predict_noise": True,
        },
        {
            "n_epochs": DIFF_EPOCHS,
            "verbose": True,
        },
        {
            "use_raw_data": True,
        },
    ),
    "DPAttentionGAN": (
            DPattentionGAN_Synthesiser,
            {
                'batch_size': 1024,
                'latent_dim': 64,
                "gen_dims": (256, 256),
                "dis_dims": (256, 256),
                'gen_lr': 0.0001,
                'dis_lr': 0.0007,
                "ae_lr": 0.02,
                "ae_compress_dim": 16,
                "ae_eps_frac": 0.3,
                'epoch_target': GAN_EPOCHS,
                'mlflow_logging': True,
            },
            {
                "n_epochs": GAN_EPOCHS,
            },
            {
                "use_raw_data": False,
            },
    ),
}


def do_things(input_dataset, output_dir="/home/azureuser/drive1/syn"):
    DIR = Path("stuff")
    HOMEDIR = os.getenv('HOME')
    DATADIR = Path(os.path.join(HOMEDIR, "TableDiffusion2/data"))
    RESULTDIR = DIR / "results"

    sys.path.append(str(SRCDIR))

    DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    SEED = 900

    for p in [SRCDIR, DIR, DATADIR, RESULTDIR]:
        if not os.path.exists(p):
            print(f"{p} does not exist")

    # Set up the models to run
    model_configs = {model_name: SYNTHESIZERS[model_name] for model_name in args.models}

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
        synthesisers=model_configs,
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
    parser.add_argument('--models', nargs='+', default=["DPWGAN", "DPDiffusion"], help='List of models to run. Options are DPWGAN, DPautoGAN, PATEGAN, DPDiffusion')
    args = parser.parse_args()

    do_things(args.input_dataset, args.output_dir)