from typing import Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
from . import core_models, cpd_models as models
from .core_models import fix_seeds

class TransposeLast2D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(-1, -2)

def str2tuple(str_value: Optional[str]) -> Optional[Union[Tuple[int], int]]:
    if str_value is not None:
        output_tuple = tuple(map(int,str_value.split(",")))
        if len(output_tuple) == 1:
            output_tuple = output_tuple[0]
    else:
        output_tuple = None

    return output_tuple

default_values = {
    "tl": {
        "rnn_hid_dim": (16, 4, 4),
        "emb_dim": (32, 8, 8)
    },
    "lin": {
        "rnn_hid_dim": 16,
        "emb_dim": 100
    },
    # "trl": {
    #     "ranks_input": (32, 4, 4, 8, 4, 4),
    #     "ranks_output": (8, 4, 4, 32, 4, 4),
    #     "ranks_rnn": (8, 4, 4, 8, 4, 4)
    # },
    "trl-half": {
        "ranks_input": (192, 8, 8),
        "ranks_output": (8, 4, 4),
        "ranks_rnn": (8, 4, 4)
    },
    "trl3dhalf": {
        "ranks_input": (192, 8, 8),
        "ranks_output": (8, 4, 4),
        "ranks_rnn": (8, 4, 4)
    },
    "tt": {
        "ranks_input": (32, 32),
        "ranks_output": (8, 8),
        "ranks_rnn": (16, 16)
    }
}

def check_default(args: Dict, block_type: str, keys: Union[str, Sequence[str]]) -> None:

    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        if args[key] is None:
            args[key] = default_values[block_type][key]


def get_args(parser):

    args_local = parser.parse_args()

    args_local_dict = dict(vars(args_local))
    args = {}
    copy_keys = [
        "model", "seed", "experiments_name", "epochs", "lr", "dryrun", "data_dim",
        "bias_rank", "block_type", "rnn_type", "input_block", "output_block"
    ]

    for key in copy_keys:
        args[key] = args_local_dict[key]

    args["bias"] = "all"
    args['num_layers'] = 1
    args["name"] = args_local.ext_name

    # if args["bias_rank"] == -1:
    #     args["bias_rank"] = "full"
    if args["output_block"] == "same":
        args["output_block"] = args["block_type"]

    if not args["experiments_name"].startswith("synthetic"):
        if  args["block_type"].startswith("linear") and \
            args["input_block"] in ["none", "linear"]:

            args['data_dim'] = 12288
        else:
            args['data_dim'] = (192, 8, 8)
    else:
        assert args['data_dim'] == args["experiments_name"].split("_")[1][:-1], \
            f'Cannot match D for synthetic dataset: {args["data_dim"]} != {args["experiments_name"]}'
        args['data_dim'] = int(args['data_dim'])
    
    args['rnn_hid_dim'] = str2tuple(args_local.hid_dim)
    args['emb_dim'] = str2tuple(args_local.emb_dim)
    lin_types = ["linear", "masked"]
    block_type = "lin" if args["block_type"] in lin_types else "tl"
    check_default(args, block_type, ["rnn_hid_dim", "emb_dim"])


    if args["block_type"] in ["trl", "trl-half", "trl3dhalf", "tt"]:
        args['ranks_input'] = str2tuple(args_local.input_ranks)
        args['ranks_rnn'] = str2tuple(args_local.rnn_ranks)
        args['ranks_output'] = str2tuple(args_local.output_ranks)
        check_default(args, args["block_type"],
                      ["ranks_input", "ranks_rnn", "ranks_output"])

    # FIXME why do we set to None?
    args['ranks_input'] = None
    args['ranks_rnn'] = None
    args['ranks_output'] = None
    if args["input_block"] in ["trl", "trl-half", "trl3dhalf", "tt"]:
        args['ranks_input'] = str2tuple(args_local.input_ranks)
        check_default(args, args["input_block"],"ranks_input")

    if args["block_type"] in ["trl", "trl-half", "trl3dhalf", "tt"]:
        args['ranks_rnn'] = str2tuple(args_local.rnn_ranks)
        check_default(args, args["block_type"], "ranks_rnn")

    if args["output_block"] in ["trl", "trl-half", "trl3dhalf", "tt"]:
        args['ranks_output'] = str2tuple(args_local.output_ranks)
        check_default(args, args["output_block"], "ranks_output")

    elif args["block_type"] == "masked":
        # For Linear
        args["alphaD"] = 1e-3
        args["alphaG"] = 0.


    if args_local.model == "bce":
        args.update(get_bce_args(args_local))

    elif args_local.model == "kl-cpd":
        args.update(get_kl_cpd_args(args_local))

    # print(f'Args: {args}')
    return args

def get_bce_args(args_local):

    assert args_local.block_type in ["tcl3d", "trl3dhalf", "linear", "linear_norm", "tcl", "trl-half", "tt"], \
        f'BCE model supports only tcl*, trl* and linear blocks'

    args = {}
    args['batch_size'] = 16
    args['grad_clip'] = 0.0

    args["EarlyStoppingParameters"] = dict(monitor="val_loss",
                                           patience=args_local.patience,
                                           verbose=True,
                                           mode="min")

    return args


def get_kl_cpd_args(args_local):

    assert args_local.block_type in [
        "tcl3d", "tcl", "trl", "linear", "trl-half", "trl3dhalf", "masked", "tt"
    ], f'KL-CPD model supports only these blocks'


    args = {}
    # FIXME  add seed to config
    args['batch_size'] = 8

    args['wnd_dim'] = 4  # 8
    args['weight_decay'] = 0.
    args['grad_clip'] = 10
    args['CRITIC_ITERS'] = 5
    args['weight_clip'] = .1
    args['lambda_ae'] = 0.1  #0.001
    args['lambda_real'] = 10  #0.1
    args['window_1'] = 4  # 8
    args['window_2'] = 4  # 8
    args['sqdist'] = 50

    args["EarlyStoppingParameters"] = dict(monitor="val_mmd2_real_D",
                                           stopping_threshold=1e-5,
                                           verbose=True,
                                           mode="min",
                                           patience=args_local.patience)


    if args_local.block_type in ["trl", "trl-half", "trl3dhalf", "tt"]:
        args['ranks_input'] = str2tuple(args_local.input_ranks)
        args['ranks_output'] = str2tuple(args_local.output_ranks)
        check_default(args, args_local.block_type, ["ranks_input", "ranks_output"])
    return args


def count_params(model):

    counter = 0
    for param in model.parameters():
        counter += param.numel()

    return counter


def get_model(args,
              train_dataset,
              test_dataset,
              extractor=None,
              freeze_extractor=True,
              add_param_numel_to_args=True):

    if extractor is None:
        print(f'Use extractor {args["name"]}')
        extractor = torch.hub.load('facebookresearch/pytorchvideo:main',
                                   args["name"],
                                   pretrained=True)
        extractor = torch.nn.Sequential(*list(extractor.blocks[:5]))

    if args["experiments_name"].startswith("synthetic"):
        # NOTE reshape to fix video extractor
        extractor = TransposeLast2D()

    if add_param_numel_to_args:
        args["extractor_params"] = count_params(extractor)

    if args["model"] == "bce":
        model = get_bce_model(args, extractor, train_dataset, test_dataset)
        if add_param_numel_to_args:
            args["model_params"] = count_params(model.model)

    elif args["model"] == "kl-cpd":
        model = get_kl_cpd_model(args, extractor, train_dataset, test_dataset)
        if add_param_numel_to_args:
            args["model_gen_params"] = count_params(model.net_generator)
            args["model_disc_params"] = count_params(model.net_discriminator)
    else:
        raise ValueError(f'Unknown model {args["model"]}')

    if freeze_extractor:
        for param in model.extractor.parameters():
            param.requires_grad = False

    return model


def get_bce_model(args, extractor, train_dataset, test_dataset):
    core_model = core_models.BceRNNTl_v2(args)

    model = models.CPDModel(core_model,
                            args,
                            train_dataset=train_dataset,
                            test_dataset=test_dataset,
                            extractor=extractor)

    return model


def get_kl_cpd_model(args, extractor, train_dataset, test_dataset):

    if args["block_type"] == "linear":
        net_gen = core_models.KlcpdGen(args)
        net_disc = core_models.KlcpdDisc(args)
    elif args["block_type"] == "masked":
        raise ValueError("Not implemented yet")
        #netG = core_models.NetG_Masked(args)
        #netD = core_models.NetD_Masked(args)
    else:
        net_gen = core_models.KlcpdGenTl(args)
        net_disc = core_models.KlcpdDiscTl(args)

    model = models.KLCPDVideo(net_gen,
                              net_disc,
                              args,
                              train_dataset=train_dataset,
                              test_dataset=test_dataset,
                              extractor=extractor)

    return model


#####
