from typing import Dict
from utils import datasets, model_utils, metrics

import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import argparse
import pickle

os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
parser.add_argument("--model", type=str, required=True, help='model name', 
                    choices=["bce", "kl-cpd"])
parser.add_argument("--ext-name", type=str, default="x3d_m", 
                    help='name of extractor model')
parser.add_argument("-tn", "--threshold-number", type=int, default=5, 
                    help='threshold number')
parser.add_argument("--experiments-name", type=str, help='name of dataset', 
                    choices=["explosion", "road_accidents"])
parser.add_argument("--scales", type=int, nargs='*', 
                    help='scales for `get_klcpd_output`')
args_local = parser.parse_args()

if args_local.model == "bce":
    if args_local.scales is not None:
        raise ValueError("BCE model has no scales")
    else:
        args_local.scales = ["none"]
elif args_local.model == "kl-cpd" and args_local.scales is None:
    args_local.scales = scales_part

experiments_name = args_local.experiments_name
train_dataset, test_dataset = datasets.CPDDatasets(experiments_name=experiments_name).get_dataset_()

name = args_local.ext_name
timestamp = args_local.timestamp
model_name = f'{name}_{args_local.model}_tl_{timestamp}'
save_path = Path("saves/models") / experiments_name / f'model_{model_name}.pth'
assert save_path.exists(), f"Checkpoint {str(save_path)} doesn't exist"

checkpoint = torch.load(save_path)
state_dict = checkpoint["checkpoint"]
args = checkpoint["args"]
# block_type = args["block_type"]
# bias = args["bias"]

seed = 0 # args["seed"]
model_utils.fix_seeds(seed)

model = model_utils.get_model(args, train_dataset, test_dataset)
model.load_state_dict(state_dict)

threshold_number = args_local.threshold_number # 25
threshold_list = np.linspace(-5, 5, threshold_number)
threshold_list = 1 / (1 + np.exp(-threshold_list))
threshold_list = [-0.001] + list(threshold_list) + [1.001]

model_type = "klcpd" if args_local.model == "kl-cpd" else "seq2seq"

metrics_local, delay_list2d, fp_delay_list2d = \
    metrics.evaluation_pipeline(model,
                                model.val_dataloader(),
                                threshold_list,
                                device='cuda',
                                model_type=model_type,
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
