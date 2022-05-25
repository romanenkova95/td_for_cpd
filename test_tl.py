from utils import datasets, kl_cpd, models_v2 as models, nets_tl, nets_original, metrics

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import matplotlib.pyplot as plt
import numpy as np

import os
import argparse

parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument("timestamp", type=int, help='timestamp to be processed')
parser.add_argument("--ext-name", type=str, default="x3d_m", help='name of extractor model')
parser.add_argument("-tn", "--threshold-number", type=int, default=5, help='threshold number')
args_local = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

experiments_name = 'explosion'
train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

name = args_local.ext_name
timestamp = args_local.timestamp
checkpoint = torch.load(f'saves/models/model_{name}_tl_{timestamp}.pth')    
state_dict = checkpoint["checkpoint"]
args = checkpoint["args"]
block_type = args["block_type"]
bias = args["bias"]
if "name" not in args:
    args["name"] = args_local.ext_name


seed = 0 # args["seed"]
models.fix_seeds(seed)
experiments_name = ('explosion')
    
if block_type == "linear":
    netG = nets_original.NetG(args)
    netD = nets_original.NetD(args)
elif args["block_type"] == "masked":
    netG = nets_original.NetG_Masked(args)
    netD = nets_original.NetD_Masked(args)
else:
    netG = nets_tl.NetG_TL(args, block_type=block_type, bias=bias)
    netD = nets_tl.NetD_TL(args, block_type=block_type, bias=bias)

extractor = torch.hub.load('facebookresearch/pytorchvideo:main', args["name"], pretrained=True)
extractor = torch.nn.Sequential(*list(extractor.blocks[:5]))

kl_cpd_model = models.KLCPDVideo(netG, netD, args, train_dataset=train_dataset, test_dataset=test_dataset, extractor=extractor)
kl_cpd_model.load_state_dict(state_dict)


threshold_number = args_local.threshold_number # 25
threshold_list = np.linspace(-5, 5, threshold_number)
threshold_list = 1 / (1 + np.exp(-threshold_list))
threshold_list = [-0.001] + list(threshold_list) + [1.001]

_, delay_list, fp_delay_list = metrics.evaluation_pipeline(kl_cpd_model, 
                                                           kl_cpd_model.val_dataloader(),  
                                                           threshold_list, 
                                                           device='cuda', 
                                                           model_type='klcpd',
                                                           verbose=False)    


path_to_saves = 'saves/results/'
metrics.write_metrics_to_file(f'{path_to_saves}metrics_{name}_tl_{timestamp}.txt', _, f'{name} tn {threshold_number}')    

plt.figure(figsize=(12, 12))
plt.plot(fp_delay_list.values(), delay_list.values(), '-o', markersize=8, label='TSCP')
plt.xlabel('Mean Time to False Alarm', fontsize=28)
plt.ylabel('Mean Detection Delay', fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc='upper left', fontsize=26);        
plt.savefig(f'{path_to_saves}figure_{name}_tl_{timestamp}.png', dpi=300)
