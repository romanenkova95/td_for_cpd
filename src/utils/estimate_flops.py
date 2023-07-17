"""
This script estimates number of FLOPs for given replacement config
"""
import torch as T
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis
from fvcore.nn.jit_handles import get_shape
from typing import List, Any, Dict
from collections import Counter
import numpy as np


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


def estimate_model_flops(model: nn.Module,
                         inputs: T.Tensor,
                         with_add_mul: bool): # -> Union[int, Any]:
    if with_add_mul:
        custom_ops: Dict[str, Any] = {
            "aten::mul": fvcore_mul_flop_jit,
            "aten::add": fvcore_add_flop_jit,
            "aten::add_": fvcore_add_flop_jit
        }
    else:
        custom_ops = {}

    flops_compressed = FlopCountAnalysis(model,
                                         inputs).set_op_handle(**custom_ops)
    flops_compressed_total = flops_compressed.total()
    fbmo = flops_compressed.by_module_and_operator()

    if with_add_mul:
        fbmo["prefix"] = "With add/mul"
    else:
        fbmo["prefix"] = "Without add/mul"

    return flops_compressed_total, fbmo

def calculate_model_parameters(model_orig):
    params_model = sum([param.numel() 
                        for param in model_orig.model.parameters()])
    params_extractor = sum([param.numel() 
                            for param in model_orig.extractor.parameters()])
    
    return params_model, params_extractor



def main(args):

    img_size = (args.batch_size, 3, 16, 256, 256)
    inputs = T.randn(img_size).to(device)


    flops_orig_total, fbmo = estimate_model_flops(model_orig, inputs, custom_ops)    
    print(prefix,
          f'total FLOPs {flops_orig_total}\n'
          f'model: {fbmo["model"]} ({sum(fbmo["model"].values())}, {sum(fbmo["model"].values()) / flops_orig_total * 100:.3f}%)\n'
          f'\tin: {fbmo["model.input_layer"]}\n'
          f'\trnn: {fbmo["model.rnn"]}\n'
          f'\tout: {fbmo["model.output_layer"]}\n'
          f'extractor: {fbmo["extractor"]} ({sum(fbmo["extractor"].values())}, {sum(fbmo["extractor"].values()) / flops_orig_total * 100:.3f}%)')

    params_model = sum([param.numel() 
                        for param in model_orig.model.parameters()])
    params_extractor = sum([param.numel() 
                            for param in model_orig.extractor.parameters()])
    print(f'total parameters {params_model + params_extractor}: '
          f'model {params_model}, extractor {params_extractor}')
