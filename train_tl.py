from utils import datasets, kl_cpd, models_v2 as models, nets_tl, nets_original

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch

import os
import time
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse

parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument("--ext-name", type=str, default="x3d_m", help='name of extractor model')
parser.add_argument("--block-type", type=str, default="tcl3d", help='type of block', 
                    choices=["tcl3d", "tcl", "trl", "linear", "trl-half", "masked"])
parser.add_argument("--epochs", type=int, default=200, help='Max number of epochs to train')
parser.add_argument("--bias-rank", type=int, default=4, help='bias rank in TCL')
parser.add_argument("--emb-dim", type=str, default="32,8,8", help='GRU embedding dim')
parser.add_argument("--hid-dim", type=str, default="16,4,4", help='GRU hidden dim')
parser.add_argument("--dryrun", action="store_true", help="Make test run")
args_local = parser.parse_args()


experiments_name = 'explosion'
train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

args = {}

bias = "all"

args["seed"] = 102
args["block_type"] = args_local.block_type
args["bias"] = bias
args["epochs"] = args_local.epochs# if not args_local.dryrun else 1
args["name"] = args_local.ext_name

args['wnd_dim'] = 4 # 8
args['batch_size'] = 8
args['lr'] = 1e-4
args['weight_decay'] = 0.
args['grad_clip'] = 10
args['CRITIC_ITERS'] = 5
args['weight_clip'] = .1
args['lambda_ae'] = 0.1 #0.001
args['lambda_real'] = 10 #0.1
args['num_layers'] = 1
args['window_1'] = 4 # 8
args['window_2'] = 4 # 8
args['sqdist'] = 50

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
    args['RNN_hid_dim'] = 16
    args['emb_dim'] = 100

elif args["block_type"] == "masked":
    # For Linear
    args['data_dim'] = 12288
    args['RNN_hid_dim'] = 16
    args['emb_dim'] = 100
    args["alphaD"] = 1e-3
    args["alphaG"] = 0.

print(f'Args: {args}')

seed = args["seed"]
models.fix_seeds(seed)
experiments_name = ('explosion')
    
if args["block_type"] == "linear":
    netG = nets_original.NetG(args)
    netD = nets_original.NetD(args)
elif args["block_type"] == "masked":
    netG = nets_original.NetG_Masked(args)
    netD = nets_original.NetD_Masked(args)
else:
    netG = nets_tl.NetG_TL(args, block_type=args["block_type"], bias=args["bias"])
    netD = nets_tl.NetD_TL(args, block_type=args["block_type"], bias=args["bias"])

print(f'Use extractor {args["name"]}')
extractor = torch.hub.load('facebookresearch/pytorchvideo:main', args["name"], pretrained=True)
extractor = torch.nn.Sequential(*list(extractor.blocks[:5]))

kl_cpd_model = models.KLCPDVideo(netG, netD, args, train_dataset=train_dataset, test_dataset=test_dataset, extractor=extractor)


logger = TensorBoardLogger(save_dir='logs/explosion', name='kl_cpd')
early_stop_callback = EarlyStopping(monitor="val_mmd2_real_D", stopping_threshold=1e-5, 
                                    verbose=True, mode="min", patience=5)



for param in kl_cpd_model.extractor.parameters():
    param.requires_grad = False

trainer = pl.Trainer(
    max_epochs=args["epochs"], # 100
    gpus='1',
    # devices='1',
    benchmark=True,
    check_val_every_n_epoch=1,
    gradient_clip_val=args['grad_clip'],
    logger=logger,
    callbacks=early_stop_callback
)

trainer.fit(kl_cpd_model)

if not args_local.dryrun:
    torch.save({"checkpoint": kl_cpd_model.state_dict(), "args": args}, f'saves/models/model_{args["name"]}_tl_{int(time.time())}.pth')    
