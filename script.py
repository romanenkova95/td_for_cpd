from utils import datasets, model_utils

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pathlib import Path
from datetime import datetime

import torch

import os
import time
import sys
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='Test your model')
    parser.add_argument("--model", type=str, required=True, help='model name', choices=["bce", "kl-cpd"])
    parser.add_argument("--ext-name", type=str, default="x3d_m", help='name of extractor model')
    parser.add_argument("--block-type", type=str, help='type of block',
                        choices=["tcl3d", "tcl", "trl", "linear", "trl-half", "trl3dhalf", "masked"])
    parser.add_argument("--rnn-type", type=str, help='type of rnn',
                        choices=["gru", "lstm"])
    parser.add_argument("--flatten-type", type=str, default="none", help='type of flatten',
                        choices=["none", "trl"])
    parser.add_argument("--epochs", type=int, default=200, help='Max number of epochs to train')
    parser.add_argument("--bias-rank", type=int, default=4, help='bias rank in TCL')
    parser.add_argument("--emb-dim", type=str, help='GRU embedding dim')
    parser.add_argument("--hid-dim", type=str, help='GRU hidden dim')
    parser.add_argument("--input-ranks", type=str, help='GRU hidden dim')
    parser.add_argument("--output-ranks", type=str, help='GRU hidden dim')
    parser.add_argument("--rnn_ranks", type=str, help='GRU hidden dim')
    parser.add_argument("--dryrun", action="store_true", help="Make test run")
    parser.add_argument("--experiments-name", type=str, default="road_accidents", help='name of dataset', choices=["explosion", "road_accidents"])
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=102, help="Random seed")
    return parser



def main(args):

    experiments_name = args["experiments_name"]
    timestamp = ""
    if not args["dryrun"]:
        timestamp = datetime.now().strftime("%y%m%dT%H%M%S")
        save_path = Path("saves/models") / experiments_name
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'model_{args["name"]}_{args["model"]}_tl_{timestamp}.pth'
        assert not save_path.exists(), f'Checkpoint {str(save_path)} already exists'

    train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

    seed = args["seed"]
    model_utils.fix_seeds(seed)

    model = model_utils.get_model(args, train_dataset, test_dataset)

    logger = TensorBoardLogger(save_dir=f'logs/{experiments_name}',
                               name=args["model"])

    trainer = pl.Trainer(
        max_epochs=args["epochs"], # 100
        gpus='1',
        benchmark=True,
        check_val_every_n_epoch=1,
        gradient_clip_val=args['grad_clip'],
        logger=logger,
        callbacks=EarlyStopping(**args["EarlyStoppingParameters"])
    )

    trainer.fit(model)

    if not args["dryrun"]:
        esp = args.pop("EarlyStoppingParameters", {})
        esp = {f'early_stopping_{key}': value for key, value in esp.items()}
        args.update(esp)
        torch.save({"checkpoint": model.state_dict(), "args": args}, save_path)

    return timestamp

if __name__ == '__main__':

    parser = get_parser()
    args = model_utils.get_args(parser)
    timestamp = main(args)
    print(timestamp)
