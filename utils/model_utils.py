import torch
import nets_original
import nets_tl
import models_v2 as models
from models_v2 import fix_seeds

def get_args(parser):

    args_local = parser.parse_args()

    args = {}
    args["seed"] = args_local.seed
    args["experiments_name"] = args_local.experiments_name

    args["name"] = args_local.ext_name
    args["epochs"] = args_local.epochs
    args['lr'] = args_local.lr


    args["dryrun"] = args_local.dryrun
    if args_local.bias_rank == -1:
        args_local.bias_rank = "full"

    args['bias_rank'] = args_local.bias_rank

    if args_local.model == "bce":
        args.update(get_bce_args(args_local))

    elif args_local.model == "kl-cpd":
        args.update(get_kl_cpd_args(args_local))

    # print(f'Args: {args}')
    return args

def get_bce_args(args_local):

    assert args_local.block_type in ["tcl3d", "linear"], \
        f'BCE model supports only tcl3d and linear blocks'

    args = {}
    args['batch_size'] = 16
    args['grad_clip'] = 0.0
    args['num_layers'] = 1

    args["block_type"] = args_local.block_type

    if args["block_type"] == "linear":
        args['data_dim'] = 12288

    elif args["block_type"] == "tcl3d":
        args['data_dim'] = (192, 8, 8)


    if args["block_type"] == "tcl3d":
        args['RNN_hid_dim'] = tuple([
            int(i) for i in args_local.hid_dim.split(",")
        ]) if args_local.hid_dim is not None else (32, 8, 8)
        args['emb_dim'] = tuple([
            int(i) for i in args_local.emb_dim.split(",")
        ]) if args_local.emb_dim is not None else (64, 8, 8)

    elif args["block_type"] == "linear":
        args['RNN_hid_dim'] = \
            int(args_local.hid_dim) if args_local.hid_dim is not None else 64
        args['emb_dim'] = \
            int(args_local.emb_dim) if args_local.emb_dim is not None else 12288

    args["EarlyStoppingParameters"] = dict(monitor="val_loss",
                                           patience=args_local.patience,
                                           verbose=True,
                                           mode="min")

    return args


def get_kl_cpd_args(args_local):

    assert args_local.block_type in [
        "tcl3d", "tcl", "trl", "linear", "trl-half", "masked"
    ], f'KL-CPD model supports only these blocks'

    args = {}
    # FIXME  add seed to config
    args["block_type"] = args_local.block_type
    args["bias"] = "all"
    args['batch_size'] = 8

    args['wnd_dim'] = 4  # 8
    args['weight_decay'] = 0.
    args['grad_clip'] = 10
    args['CRITIC_ITERS'] = 5
    args['weight_clip'] = .1
    args['lambda_ae'] = 0.1  #0.001
    args['lambda_real'] = 10  #0.1
    args['num_layers'] = 1
    args['window_1'] = 4  # 8
    args['window_2'] = 4  # 8
    args['sqdist'] = 50

    if args["block_type"] in ["tcl3d", "tcl", "trl", "trl-half"]:
        args['data_dim'] = (192, 8, 8)

    elif args["block_type"] in ["linear", "masked"]:
        args['data_dim'] = 12288


    if args["block_type"] in ["tcl3d", "tcl", "trl", "trl-half"]:

        args['RNN_hid_dim'] = tuple([
            int(i) for i in args_local.hid_dim.split(",")
        ]) if args_local.hid_dim is not None else (16, 4, 4)
        args['emb_dim'] = tuple([
            int(i) for i in args_local.emb_dim.split(",")
        ]) if args_local.emb_dim is not None else (32, 8, 8)

    elif args["block_type"] in ["linear", "masked"]:
        args['RNN_hid_dim'] = \
            int(args_local.hid_dim) if args_local.hid_dim is not None else 16
        args['emb_dim'] = \
            int(args_local.emb_dim) if args_local.emb_dim is not None else 100


    if args["block_type"] == "trl" :
        # For TRL
        args['ranks_input'] = (32, 4, 4, 8, 4, 4)  # 3072
        args['ranks_output'] = (8, 4, 4, 32, 4, 4)  # 3072
        args['ranks_gru'] = (8, 4, 4, 8, 4, 4)  # 3072
        # args['bias_rank'] = 0  #args_local.bias_rank

    elif args["block_type"] == "trl-half":
        # For TRL
        args['ranks_input'] = (32, 4, 4)
        args['ranks_output'] = (8, 4, 4)
        args['ranks_gru'] = (8, 4, 4)
        # args['bias_rank'] = 0  #args_local.bias_rank

    elif args["block_type"] == "masked":
        # For Linear
        args["alphaD"] = 1e-3
        args["alphaG"] = 0.

    args["EarlyStoppingParameters"] = dict(monitor="val_mmd2_real_D",
                                           stopping_threshold=1e-5,
                                           verbose=True,
                                           mode="min",
                                           patience=args_local.patience)

    return args


def get_model(args,
              train_dataset,
              test_dataset,
              extractor=None,
              freeze_extractor=True):

    if extractor is None:
        print(f'Use extractor {args["name"]}')
        extractor = torch.hub.load('facebookresearch/pytorchvideo:main',
                                   args["name"],
                                   pretrained=True)
        extractor = torch.nn.Sequential(*list(extractor.blocks[:5]))

    if args["model"] == "bce":
        model = get_bce_model(args, extractor, train_dataset, test_dataset)
    elif args["model"] == "bce":
        model = get_kl_cpd_model(args, extractor, train_dataset, test_dataset)
    else:
        raise ValueError(f'Unknown model {args["model"]}')

    if freeze_extractor:
        for param in model.extractor.parameters():
            param.requires_grad = False

    return model


def get_bce_model(args, extractor, train_dataset, test_dataset):

    if args["block_type"] == "linear":
        print(args)
        core_model = nets_original.BCE_GRU(args)
    else:
        core_model = nets_tl.BCE_GRU_TL(args,
                                        block_type=args["block_type"],
                                        bias=args["bias"])

    model = models.CPD_model(core_model,
                             args,
                             train_dataset=train_dataset,
                             test_dataset=test_dataset,
                             extractor=extractor,
                             batch_size=args['batch_size'])

    return model


def get_kl_cpd_model(args, extractor, train_dataset, test_dataset):

    if args["block_type"] == "linear":
        netG = nets_original.NetG(args)
        netD = nets_original.NetD(args)
    elif args["block_type"] == "masked":
        netG = nets_original.NetG_Masked(args)
        netD = nets_original.NetD_Masked(args)
    else:
        netG = nets_tl.NetG_TL(args,
                               block_type=args["block_type"],
                               bias=args["bias"])
        netD = nets_tl.NetD_TL(args,
                               block_type=args["block_type"],
                               bias=args["bias"])

    model = models.KLCPDVideo(netG,
                              netD,
                              args,
                              train_dataset=train_dataset,
                              test_dataset=test_dataset,
                              extractor=extractor)

    return model
