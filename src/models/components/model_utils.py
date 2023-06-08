from typing import Dict, Optional, Sequence, Tuple, Union
from ..src.models import core_models
import torch
import torch.nn as nn
from ..src.models import cpd_models as models
from ..src.models.core_models import fix_seeds


def get_kl_cpd_args(args_local):
    assert args_local.block_type in [
        "tcl3d",
        "tcl",
        "trl",
        "linear",
        "trl-half",
        "trl3dhalf",
        "masked",
        "tt",
    ], f"KL-CPD model supports only these blocks"

    args = {}
    # FIXME  add seed to config
    args["batch_size"] = 8

    args["wnd_dim"] = 4  # 8
    args["weight_decay"] = 0.0
    args["grad_clip"] = 10
    args["CRITIC_ITERS"] = 5
    args["weight_clip"] = 0.1
    args["lambda_ae"] = 0.1  # 0.001
    args["lambda_real"] = 10  # 0.1
    args["window_1"] = 4  # 8
    args["window_2"] = 4  # 8
    args["sqdist"] = 50

    args["EarlyStoppingParameters"] = dict(
        monitor="val_mmd2_real_D",
        stopping_threshold=1e-5,
        verbose=True,
        mode="min",
        patience=args_local.patience,
    )

    if args_local.block_type in ["trl", "trl-half", "trl3dhalf", "tt"]:
        args["ranks_input"] = str2tuple(args_local.input_ranks)
        args["ranks_output"] = str2tuple(args_local.output_ranks)
        check_default(args, args_local.block_type, ["ranks_input", "ranks_output"])
    return args


def count_params(model):
    counter = 0
    for param in model.parameters():
        counter += param.numel()

    return counter
