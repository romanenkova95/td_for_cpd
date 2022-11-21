from utils import datasets, kl_cpd, models_v2 as models, nets_tl, nets_original

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

bias = "all"


def get_parser():
    parser = argparse.ArgumentParser(description='Test your model')
    parser.add_argument("--ext-name", type=str, default="x3d_m", help='name of extractor model')
    parser.add_argument("--block-type", type=str, default="tcl3d", help='type of block',
                        choices=["tcl3d", "tcl", "trl", "linear", "trl-half", "masked"])
    parser.add_argument("--epochs", type=int, default=200, help='Max number of epochs to train')
    parser.add_argument("--bias-rank", type=int, default=4, help='bias rank in TCL')
    parser.add_argument("--emb-dim", type=str, default="32,8,8", help='GRU embedding dim')
    parser.add_argument("--hid-dim", type=str, default="16,4,4", help='GRU hidden dim')
    parser.add_argument("--dryrun", action="store_true", help="Make test run")
    parser.add_argument("--experiments-name", type=str, default="road_accidents", help='name of dataset', choices=["explosion", "road_accidents"])
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    return parser

def get_args(parser):

    args_local = parser.parse_args()
    args = {}
    # FIXME  add seed to config
    args["seed"] = 33
    args["experiments_name"] = args_local.experiments_name
    args["block_type"] = args_local.block_type
    args["bias"] = bias
    args["epochs"] = args_local.epochs# if not args_local.dryrun else 1
    args["name"] = args_local.ext_name
    args['patience'] = args_local.patience

    args['batch_size'] = 8
    args['lr'] = 1e-3
    args['num_layers'] = 1
    args['grad_clip'] = 0.0

    args["dryrun"] = args_local.dryrun
    if args_local.bias_rank == -1:
        args_local.bias_rank = "full"

    if args["block_type"] == "tcl3d":
        # For TCL3D
        args['data_dim'] = (192, 8, 8)
        args['RNN_hid_dim'] = tuple([int(i) for i in args_local.hid_dim.split(",")]) #(16, 4, 4) # 3072
        args['emb_dim'] = tuple([int(i) for i in args_local.emb_dim.split(",")]) #(32, 8, 8) # 3072
        args['bias_rank'] = args_local.bias_rank

    elif args["block_type"] == "tcl":
        # For TCL
        args['data_dim'] = (192, 8, 8)
        args['RNN_hid_dim'] = (64, 8, 8) # 3072
        args['emb_dim'] = (64, 8, 8) # 3072
        args["bias_rank"] = 1 # TODO CHECK!!!!!!!!!!!!!!!!!!

    elif args["block_type"] == "trl":
        # For TRL
        args['data_dim'] = (192, 8, 8)
        args['RNN_hid_dim'] = (16, 4, 4)
        args['emb_dim'] = (32, 8, 8)
        args['ranks_input'] = (32, 4, 4, 8, 4, 4) # 3072
        args['ranks_output'] = (8, 4, 4, 32, 4, 4) # 3072
        args['ranks_gru'] = (8, 4, 4, 8, 4, 4) # 3072
        args['bias_rank'] = 0 #args_local.bias_rank

    elif args["block_type"] == "trl-half":
        # For TRL
        args['data_dim'] = (192, 8, 8)
        args['RNN_hid_dim'] = (16, 4, 4)
        args['emb_dim'] = (32, 8, 8)
        args['ranks_input'] = (32, 4, 4)
        args['ranks_output'] = (8, 4, 4)
        args['ranks_gru'] = (8, 4, 4)
        args['bias_rank'] = 0 #args_local.bias_rank

    elif args["block_type"] == "linear":
        # For Linear
        args['data_dim'] = 12288
        args['RNN_hid_dim'] = 64
        args['emb_dim'] = 12288

    elif args["block_type"] == "masked":
        # For Linear
        args['data_dim'] = 12288
        args['RNN_hid_dim'] = 16
        args['emb_dim'] = 100
        args["alphaD"] = 1e-3
        args["alphaG"] = 0.

    # print(f'Args: {args}')
    return args


def main(args):

    experiments_name = args["experiments_name"]
    timestamp = ""
    if not args["dryrun"]:
        timestamp = datetime.now().strftime("%y%m%dT%H%M%S")
        save_path = Path("saves/models") / experiments_name
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f'model_{args["name"]}_bce_tl_{timestamp}.pth'
        assert not save_path.exists(), f'Checkpoint {str(save_path)} already exists'

    train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

    seed = args["seed"]
    models.fix_seeds(seed)

    if args["block_type"] == "linear":
        core_model = nets_original.BCE_GRU(args)
    elif args["block_type"] == "masked":
        pass
    else:
        core_model = nets_tl.BCE_GRU_TL(args, block_type=args["block_type"], bias=args["bias"])

    print(f'Use extractor {args["name"]}')
    extractor = torch.hub.load('facebookresearch/pytorchvideo:main', args["name"], pretrained=True)
    extractor = torch.nn.Sequential(*list(extractor.blocks[:5]))

    cpd_model = models.CPD_model(core_model,
                                 args,
                                 train_dataset=train_dataset,
                                 test_dataset=test_dataset,
                                 extractor=extractor,
                                 batch_size=args['batch_size'])

    logger = TensorBoardLogger(save_dir=f'logs/{experiments_name}', name='bce_model')
    early_stop_callback = EarlyStopping(monitor="val_loss",
                                        patience=args["patience"], verbose=True, mode="min")

    for param in cpd_model.extractor.parameters():
        param.requires_grad = False

    trainer = pl.Trainer(
        max_epochs=args["epochs"], # 100
        gpus='1',
        benchmark=True,
        check_val_every_n_epoch=1,
        gradient_clip_val=args['grad_clip'],
        logger=logger,
        callbacks=early_stop_callback
    )

    trainer.fit(cpd_model)

    if not args["dryrun"]:
        torch.save({"checkpoint": cpd_model.state_dict(), "args": args}, save_path)

    return timestamp

if __name__ == '__main__':

    parser = get_parser()
    args = get_args(parser)
    timestamp = main(args)
    print(timestamp)