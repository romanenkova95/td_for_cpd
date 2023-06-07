"""
This script estimates number of FLOPs for given replacement config
"""
from argparse import ArgumentParser
import torch as T
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape
from typing import List, Any, Dict, Tuple, Union
from collections import Counter
import numpy as np
from pathlib import Path

from utils import datasets, model_utils, metrics


def fvcore_mul_flop_jit(inputs: List[Any], outputs: List[Any]) -> Counter:
    """
    flops of aten::mul
    """
    flop_dict = Counter()
    flop_dict["mul"] = np.prod(get_shape(inputs[0]))
    return flop_dict


def fvcore_add_flop_jit(inputs: List[Any], outputs: List[Any]) -> Counter:
    """
    flops of aten::add
    """
    flop_dict = Counter()
    flop_dict["add"] = np.prod(get_shape(inputs[0]))
    return flop_dict


def get_args_parser():
    parser = ArgumentParser('DeiT training and evaluation script',
                            add_help=False)
    parser.add_argument("--timestamp", "-t", type=str, help='timestamp to be processed')
    parser.add_argument('--batch-size', "-b", default=1, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument("--with-add-mul", action='store_true')
    parser.add_argument("--model", type=str, required=True, help='model name', 
                        choices=["bce", "kl-cpd"])
    parser.add_argument("--ext-name", type=str, default="x3d_m", 
                        help='name of extractor model')
    parser.add_argument("--experiments-name", type=str, help='name of dataset', 
                        choices=["explosion", "road_accidents"])
    parser.add_argument("--not-strict", action='store_true')

    return parser


def estimate_model_flops(model: nn.Module,
                         inputs: T.Tensor,
                         custom_ops: Dict = {}): # -> Union[int, Any]:
    flops_compressed = FlopCountAnalysis(model,
                                         inputs).set_op_handle(**custom_ops)
    flops_compressed_total = flops_compressed.total()
    fbmo = flops_compressed.by_module_and_operator()

    return flops_compressed_total, fbmo

def load_model(path, strict=True):
    checkpoint = T.load(path, map_location="cpu")
    state_dict = checkpoint["checkpoint"]
    args = checkpoint["args"]
    
    model = model_utils.get_model(args, None, None)
    model.load_state_dict(state_dict, strict)
    return model

def main(args):

    device = args.device
    img_size = (args.batch_size, 3, 16, 256, 256)
    if args.with_add_mul:
        prefix = "With add/mul"
        custom_ops: Dict[str, Any] = {
            "aten::mul": fvcore_mul_flop_jit,
            "aten::add": fvcore_add_flop_jit,
            "aten::add_": fvcore_add_flop_jit
        }
    else:
        custom_ops = {}
        prefix = "Without add/mul"

    experiments_name = args.experiments_name
    name = args.ext_name
    timestamp = args.timestamp
    model_name = f'{name}_{args.model}_tl_{timestamp}'
    save_path = Path("saves/models") / experiments_name / f'model_{model_name}.pth'
    assert save_path.exists(), f"Checkpoint {str(save_path)} doesn't exist"

    model_orig = load_model(save_path, not args.not_strict).to(device)
    inputs = T.randn(img_size).to(device)

    flops_orig_total, fbmo = estimate_model_flops(model_orig, inputs, custom_ops)
    print(f'total: {flops_orig_total}')

    fbmo = flops_orig.by_module_and_operator()
    print(f'model: {fbmo["model"]} ({sum(fbmo["model"].values())}, {sum(fbmo["model"].values()) / flops_orig_total * 100:.3f}%)\n'
          f'\tin: {fbmo["model.input_layer"]}\n'
          f'\trnn: {fbmo["model.rnn"]}\n'
          f'\tout: {fbmo["model.output_layer"]}\n'
          f'extractor: {fbmo["extractor"]} ({sum(fbmo["extractor"].values())}, {sum(fbmo["extractor"].values()) / flops_orig_total * 100:.3f}%)')


if __name__ == "__main__":

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
