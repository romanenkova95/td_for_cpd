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

experiments_name = 'explosion'
train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

args = {}

block_type = "tcl3d"
bias = "all"

args["block_type"] = block_type
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

if block_type == "tcl3d":
    # For TCL3D
    args['data_dim'] = (192, 8, 8)
    args['RNN_hid_dim'] = (32, 8, 8) # 3072
    args['emb_dim'] = (64, 8, 8) # 3072
    args['bias_rank'] = 8
    
elif block_type == "tcl":
    # For TCL
    args['data_dim'] = (192, 8, 8)
    args['RNN_hid_dim'] = (64, 8, 8) # 3072
    args['emb_dim'] = (64, 8, 8) # 3072

elif block_type == "linear":
    # For Linear
    args['data_dim'] = 12288
    args['RNN_hid_dim'] = 128 # 3072
    args['emb_dim'] = 256 # 3072


seed = 0
models.fix_seeds(seed)
experiments_name = ('explosion')
    
if block_type == "linear":
    netG = nets_original.NetG(args)
    netD = nets_original.NetD(args)
else:
    netG = nets_tl.NetG_TL(args, block_type=block_type, bias=bias)
    netD = nets_tl.NetD_TL(args, block_type=block_type, bias=bias)

kl_cpd_model = models.KLCPDVideo(netG, netD, args, train_dataset=train_dataset, test_dataset=test_dataset)


logger = TensorBoardLogger(save_dir='logs/explosion', name='kl_cpd')
early_stop_callback = EarlyStopping(monitor="val_mmd2_real_D", stopping_threshold=1e-5, 
                                    verbose=True, mode="min", patience=5)



for param in kl_cpd_model.extractor.parameters():
    param.requires_grad = False

trainer = pl.Trainer(
    max_epochs=20, # 100
    gpus='1',
    # devices='1',
    benchmark=True,
    check_val_every_n_epoch=1,
    gradient_clip_val=args['grad_clip'],
    logger=logger,
    callbacks=early_stop_callback
)

trainer.fit(kl_cpd_model)
name = "x3d_m"
torch.save({"checkpoint": kl_cpd_model.state_dict(), "args": args}, f'saves/models/model_{name}_tl_{int(time.time())}.pth')    
