from utils import datasets, kl_cpd, models_v2 as models, nets_tl, nets_original, metrics

import warnings
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import os
import argparse

parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument("timestamp", type=int, help='timestamp to be processed')
parser.add_argument("--ext-name", type=str, default="x3d_m", help='name of extractor model')
parser.add_argument("-tn", "--threshold-number", type=int, default=5, help='threshold number')
parser.add_argument("--experiments-name", type=str, default="explosion", help='name of dataset', choices=["explosion", "road_accidents"])
args_local = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

experiments_name = args_local.experiments_name
train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

name = args_local.ext_name
timestamp = args_local.timestamp
model_name = f'{name}_tl_{timestamp}'
save_path = Path("saves/models") / experiments_name / f'model_{model_name}.pth'
assert save_path.exists(), f"Checkpoint {str(save_path)} doesn't exist"

checkpoint = torch.load(save_path)    
state_dict = checkpoint["checkpoint"]
args = checkpoint["args"]
block_type = args["block_type"]
bias = args["bias"]
if "name" not in args:
    args["name"] = args_local.ext_name


seed = 0 # args["seed"]
models.fix_seeds(seed)
    
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


path_to_saves = Path('saves/results') / experiments_name
path_to_metric = path_to_saves / "metrics"
path_to_metric.mkdir(parents=True, exist_ok=True)

metrics.write_metrics_to_file(f'{str(path_to_metric)}/{model_name}.txt', _, f'{name} tn {threshold_number}')    

plt.figure(figsize=(12, 12))
plt.plot(fp_delay_list.values(), delay_list.values(), '-o', markersize=8, label='TSCP')
plt.xlabel('Mean Time to False Alarm', fontsize=28)
plt.ylabel('Mean Detection Delay', fontsize=28)
plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(loc='upper left', fontsize=26);     

path_to_figure = path_to_saves / "figures" 
path_to_figure.mkdir(parents=True, exist_ok=True)
plt.savefig(path_to_figure / f'{model_name}.png', dpi=300)
