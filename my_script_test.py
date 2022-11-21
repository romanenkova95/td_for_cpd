from utils import datasets, kl_cpd, models_v2 as models, nets_tl, nets_original, metrics

import warnings
warnings.filterwarnings("ignore")

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import time
import argparse
import pickle


def dump_results(metrics_local_dict, timestamp: str):

    results = []

    for scale, metrics_local in metrics_local_dict.items():
        best_th_f1, best_time_to_FA = metrics_local
        results += [
            dict(scale=scale,
                 best_th_f1=best_th_f1,
                 best_time_to_FA=best_time_to_FA,
        )]

    with Path(f'saves/{timestamp}-temp.pickle').open("wb") as f:
        pickle.dump(results, f)


scales_full = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]
scales_part = [1e4, 1e5, 1e6, 1e7]

parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument("timestamp", type=str, help='timestamp to be processed')
parser.add_argument("--ext-name", type=str, default="x3d_m", help='name of extractor model')
parser.add_argument("-tn", "--threshold-number", type=int, default=5, help='threshold number')
parser.add_argument("--experiments-name", type=str, default="explosion", help='name of dataset', choices=["explosion", "road_accidents"])
parser.add_argument("--scales", type=int, nargs='+', default=scales_part, help='scales for `get_klcpd_output_2`')
args = parser.parse_args()


# print(args)
time.sleep(2)
dump_results({1000: (1, 2), 10000: (3, 4)}, args.timestamp)
