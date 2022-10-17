from typing import Dict
from utils import datasets, kl_cpd, models_v2 as models, nets_tl, nets_original, metrics

import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import argparse
import pickle

def dump_results(metrics_local_dict: Dict, timestamp: str):

    results = []
    for scale, metrics_local in metrics_local_dict.items():
        best_th_f1, best_time_to_FA, best_delay, auc, best_conf_matrix, best_f1, best_cover, best_th_cover, max_cover = metrics_local
        results += [
            dict(scale=scale,
                 best_th_f1=best_th_f1,
                 best_time_to_FA=best_time_to_FA,
                 best_delay=best_delay,
                 auc=auc,
                 best_conf_matrix=best_conf_matrix,
                 best_f1=best_f1,
                 best_cover=best_cover,
                 best_th_cover=best_th_cover,
                 max_cover=max_cover)
        ]

    with Path(f'saves/temp_{timestamp}.pickle').open("wb") as f:
        pickle.dump(results, f)


scales_full = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]
scales_part = [1e4, 1e5, 1e6, 1e7]

parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument("timestamp", type=str, help='timestamp to be processed')
parser.add_argument("--ext-name", type=str, default="x3d_m", help='name of extractor model')
parser.add_argument("-tn", "--threshold-number", type=int, default=5, help='threshold number')
parser.add_argument("--experiments-name", type=str, default="explosion", help='name of dataset', choices=["explosion", "road_accidents"])
parser.add_argument("--scales", type=int, nargs='+', default=scales_part, help='scales for `get_klcpd_output_2`')
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

metrics_local, delay_list2d, fp_delay_list2d = \
    metrics.evaluation_pipeline(kl_cpd_model,
                                kl_cpd_model.val_dataloader(),
                                threshold_list,
                                device='cuda',
                                model_type='klcpd',
                                verbose=False,
                                scales=args_local.scales)


path_to_saves = Path('saves/results') / experiments_name
path_to_metric = path_to_saves / "metrics"
path_to_metric.mkdir(parents=True, exist_ok=True)

metrics.write_metrics_to_file(f'{str(path_to_metric)}/{model_name}.txt',
                              metrics_local,
                              f'{name} tn {threshold_number}')

for scale, fp_delay_list, delay_list in zip(args_local.scales, fp_delay_list2d, delay_list2d):

    fp_delay_list = fp_delay_list2d[scale]
    delay_list = delay_list2d[scale]

    plt.figure(figsize=(12, 12))
    plt.plot(fp_delay_list.values(), delay_list.values(), '-o', markersize=8, label='TSCP')
    plt.xlabel('Mean Time to False Alarm', fontsize=28)
    plt.ylabel('Mean Detection Delay', fontsize=28)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(loc='upper left', fontsize=26);

    path_to_figure = path_to_saves / "figures"
    path_to_figure.mkdir(parents=True, exist_ok=True)
    plt.savefig(path_to_figure / f'{model_name}_{scale}.png', dpi=300)

dump_results(metrics_local, timestamp)
