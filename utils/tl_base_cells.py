from typing import Dict, Tuple
import torch
import torch.nn as nn
import numpy as np

from .tensor_layers import TCL, TCL3D, TRLhalf, TRL3Dhalf


class GruTl(nn.Module):
    def __init__(
        self,
            block_type: str,
            input_dims: Tuple[int, int, int],
            hidden_dims: int,
            ranks = None,
            bias_rank = False,
            batch_first=True
    ) -> object:
        super().__init__()

        self.hidden_dims = hidden_dims
        self.batch_first = batch_first

        if block_type.lower() == "tcl3d":
            block, args = TCL3D, {}
        elif block_type.lower() == "tcl":
            block, args = TCL, {}
        # elif block_type.lower() == "trl":
        #     block, args = TRL, {"core_shape": ranks}
        elif block_type.lower() == "trl-half":
            block, args = TRLhalf, {"core_shape": ranks}
        elif block_type.lower() == "trl3dhalf":
            block, args = TRL3Dhalf, {"core_shape": ranks}
        else:
            raise ValueError(
                f"Incorrect block type: {block_type}. Should be tcl or trl"
            )

        self.linear_w = nn.ModuleList(
            [
                block(
                    input_dims,
                    hidden_dims,
                    **args,
                    bias_rank=bias_rank,
                )
                for _ in range(3)
            ]
        )

        self.linear_u = nn.ModuleList(
            [
                block(
                    hidden_dims,
                    hidden_dims,
                    **args,
                    bias_rank=bias_rank,
                )
                for _ in range(3)
            ]
        )

    def forward(self, inputs, h_prev=None):

        if self.batch_first:
            # sequence first (timesteps, batch_size, input_dims)
            inputs = inputs.transpose(0, 1)

        # inputs shape: L, N, Hin
        L, N, *input_dims = inputs.shape

        outputs = []
        if h_prev is None:
            h_prev = torch.zeros(N, *self.hidden_dims).to(inputs)

        for x_t in inputs:
            x_z, x_r, x_h = [linear(x_t) for linear in self.linear_w]
            h_z, h_r, h_h = [linear(h_prev) for linear in self.linear_u]

            output_z = torch.sigmoid(x_z + h_z)
            output_r = torch.sigmoid(x_r + h_r)
            hidden_hat = torch.tanh(x_h + output_r * h_h)
            h_prev = output_z * h_prev + (1 - output_z) * hidden_hat

            outputs.append(
                h_prev
            )  # .clone().detach()) # NOTE fail with detach. #TODO check it

        outputs = torch.stack(outputs, dim=0)

        if self.batch_first:
            # batch first (batch_size, timesteps, input_dims)
            outputs = outputs.transpose(0, 1)

        return outputs, h_prev


class LstmTl(nn.Module):

    # TODO add num_layers parameter
    def __init__(self, block_type, input_dims, hidden_dims, ranks=None, bias_rank=False, batch_first=True):
        super().__init__()

        self.hidden_dims = hidden_dims
        self.batch_first = batch_first

        if block_type.lower() == "tcl3d":
            block, args = TCL3D, {}
        elif block_type.lower() == "tcl":
            block, args = TCL, {}
        # elif block_type.lower() == "trl":
        #     block, args = TRL, {"core_shape": ranks}
        elif block_type.lower() == "trl-half":
            block, args = TRLhalf, {"core_shape": ranks}
        elif block_type.lower() == "trl3dhalf":
            block, args = TRL3Dhalf, {"core_shape": ranks}
        else:
            raise ValueError(f'Incorrect block type: {block_type}. Should be tcl or trl-half')

        self.linear_w = nn.ModuleList([
            block(input_dims, hidden_dims, **args, bias_rank=bias_rank)
            for _ in range(4)])

        self.linear_u = nn.ModuleList([
            block(hidden_dims, hidden_dims, **args, bias_rank=bias_rank)
            for _ in range(4)])


    def forward(self, inputs, init_states=None):

        if self.batch_first:
            # sequence first (timesteps, batch_size, input_dims)
            inputs = inputs.transpose(0, 1)

        # inputs shape: L, N, Hin
        L, N, *input_dims = inputs.shape

        outputs = []
        if init_states is None:
            h_prev = torch.zeros(N, *self.hidden_dims).to(inputs)
            c_prev = torch.zeros(N, *self.hidden_dims).to(inputs)
        else:
            h_prev, c_prev = init_states

        for x_t in inputs:
            # print(self.linear_u)
            x_z, x_r, x_h, x_c = [linear(x_t) for linear in self.linear_w]
            h_z, h_r, h_h, h_c = [linear(h_prev) for linear in self.linear_u]

            output_z = torch.sigmoid(x_z + h_z) # update gate
            output_r = torch.sigmoid(x_r + h_r) # forget gate
            output_o = torch.sigmoid(x_h + h_h) # output gate
            output_c = torch.sigmoid(x_r + h_r) # cell input activation vector

            c_prev = output_r * c_prev + output_z * output_c
            h_prev = output_o * torch.sigmoid(c_prev)

            outputs.append(h_prev)#.clone().detach()) # NOTE fail with detach. check it

        outputs = torch.stack(outputs, dim=0)
        # print(f'h1: {h_prev.shape}')

        if self.batch_first:
            # batch first (batch_size, timesteps, input_dims)
            outputs = outputs.transpose(0, 1)

        return outputs, (h_prev, c_prev)


def init_bias(args):
    fc_bias = args["bias_rank"] if args["bias"] in ["all", "emb_layer"] else 0
    rnn_bias = args["bias_rank"] if args["bias"] in ["all", "gru"] else 0
    return fc_bias, rnn_bias


def init_fc_rnn_tl(block_type, args):

    args_in, args_rnn, args_out = {}, {}, {}

    if block_type == "tcl3d":
        block = TCL3D
    elif block_type == "tcl":
        block = TCL
    # elif block_type == "trl":
    #     block = TRL
    elif block_type == "trl-half":
        block = TRLhalf
    elif block_type == "trl3dhalf":
        block = TRL3Dhalf
    else:
        raise ValueError(
            f'Incorrect block type: {block_type}. Should be tcl or trl-half')

    if block_type in ["trl-half", "trl3dhalf"]:
        args_in["core_shape"] = args["ranks_input"]
        args_rnn["ranks"] = args["ranks_rnn"]
        args_out["core_shape"] = args["ranks_output"]

    return block, (args_in, args_rnn, args_out)


def init_block(block_type, ranks=None, for_rnn=False):

    args = {}

    if block_type == "tcl3d":
        block = TCL3D
    elif block_type == "tcl":
        block = TCL
    # elif block_type == "trl":
    #     block = TRL
    elif block_type == "trl-half":
        block = TRLhalf
    elif block_type == "trl3dhalf":
        block = TRL3Dhalf
    else:
        raise ValueError(
            f'Incorrect block type: {block_type}. Should be tcl or trl-half')

    if block_type in ["trl-half", "trl3dhalf"]:
        assert ranks is not None, f"Ranks not provided"
        key = "ranks" if for_rnn else "core_shape"
        args[key] = ranks

    return block, args


def parse_bce_args(args: Dict):

    block_type = args["block_type"].lower()

    if block_type in ["linear"]:
        layer_input, layer_rnn, layer_output = parse_bce_linear(args)
    else:
        layer_input, layer_rnn, layer_output = parse_bce_tl(args)

    return layer_input, layer_rnn, layer_output


def parse_bce_linear(args: Dict):
    fc_bias, gru_bias = init_bias(args)
    gru_bias = isinstance(gru_bias, int) and gru_bias > 0 or gru_bias == "full"
    fc_bias_flag = isinstance(fc_bias, int) and fc_bias > 0 or fc_bias == "full"

    has_input_block = args["input_block"] not in ["flatten", "none"]
    input_dim = args['data_dim'] if not has_input_block else args['emb_dim']

    block_rnn = nn.LSTM if args["rnn_type"].lower() in ["lstm"] else nn.GRU
    layer_rnn = block_rnn(input_size=input_dim,
                          hidden_size=args['rnn_hid_dim'],
                          num_layers=args['num_layers'],
                          batch_first=True,
                          bias=gru_bias)
    layer_output = nn.Linear(in_features=args['rnn_hid_dim'],
                             out_features=1,
                             bias=fc_bias_flag)

    if args["input_block"] in ["flatten", "none"]:
        layer_input = nn.Flatten(start_dim=2)
    elif args["input_block"] == "linear":
        layer_input = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(args['data_dim'], args['emb_dim'], bias=fc_bias_flag),
            nn.ReLU())
    elif args["input_block"] in ["trl3dhalf", "trl-half"]:
        layer_input = TRLhalf(input_shape=args['data_dim'],
                                output_shape=(args['emb_dim'],),
                                core_shape=args['data_dim'],
                                bias_rank=fc_bias,
                                normalize="both")
        print(layer_input)
    else:
        raise ValueError(f'Incorrect block type: {args["input_block"]}. '
                         f'Should be flatten, none, linear or trl3dhalf')
    return layer_input, layer_rnn, layer_output


def parse_bce_tl(args: Dict):

    block_type = args["block_type"].lower()
    fc_bias, gru_bias = init_bias(args)

    has_input_block = args["input_block"] != "none"
    input_dim = args['data_dim'] if not has_input_block else args['emb_dim']
    args_rnn = {
        "block_type": block_type,
        "input_dims": input_dim,
        "hidden_dims": args['rnn_hid_dim'],
        "bias_rank": gru_bias,
    }
    block_rnn = LstmTl if args["rnn_type"].lower() in ["lstm"] else GruTl
    _, args_rnn2 = init_block(block_type, args["ranks_rnn"], for_rnn=True)
    layer_rnn = block_rnn(**args_rnn, **args_rnn2)

    if args["output_block"] != "linear":
        if args["output_block"].startswith("tcl"):
            output_shape = (1, 1, 1)
        else: # args["output_block"] == "trl3dhalf"
            output_shape = (1,)
        args_out = {
            "input_shape": args['rnn_hid_dim'],
            "output_shape": output_shape,
            "bias_rank": fc_bias,
            "normalize": "in"
        }

        block_output, args_out2 = init_block(args["output_block"],
                                            args["ranks_output"])

        layer_output = block_output(**args_out, **args_out2)
    else:
        fc_bias = isinstance(fc_bias, int) and fc_bias > 0 or fc_bias == "full"
        layer_output = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(in_features=np.prod(args['rnn_hid_dim']),
                      out_features=1,
                      bias=fc_bias))
    if has_input_block:
        args_in = {
            "input_shape": args['data_dim'],
            "output_shape": args['emb_dim'],
            "bias_rank": fc_bias,
            "normalize": "both"
        }
        block_input, args_in2 = init_block(args["input_block"],
                                           args["ranks_input"])
        layer_input = nn.Sequential(block_input(**args_in, **args_in2),
                                    nn.ReLU())
    else:
        layer_input = nn.Identity()

    return layer_input, layer_rnn, layer_output