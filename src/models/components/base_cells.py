from typing import Tuple, Union, Optional
import torch
import torch.nn as nn
import numpy as np

from tensor_layers import TCL, TCL3D, TRLhalf, TRL3Dhalf, TT
import warnings


class CoreBlock(nn.Module):
    def __init__(
        self,
        block_type: str,
        input_dims: Union[Tuple[int, int, int], int],
        hidden_dims: Union[Tuple[int, int, int], Tuple[int], int],
        ranks: Optional[Union[Tuple[int, int, int], int]] = None,
        bias_rank: Optional[int] = -1,
        normalize: str = "both",
        block_place: str = "inter",  # "input", "output", "inter" aka "intermediate"
    ) -> None:
        super().__init__()

        self.block_type = block_type.lower()
        self.block_place = block_place

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.bias_rank = bias_rank

        self._sanity_check(ranks)

        _tensor_args = {"bias_rank": bias_rank, "normalize": normalize}

        block, args = self._init_block(_tensor_args, ranks)
        self.args = args

        if self.block_place == "inter":
            if self.block_type != "flatten":
                self.block = block(self.input_dims, self.hidden_dims, **self.args)
            elif self.block_type == "flatten":
                self.block = nn.Flatten(**self.args)
        else:
            self.block = self._process_special_block(block)

    def forward(self, inputs):
        return self.block(inputs)

    def _init_block(self, tensor_args, ranks):
        # TODO: update python version till 3.10 and rewrite to match/case
        if self.block_type == "tcl3d":
            block, args = TCL3D, tensor_args
        elif self.block_type == "tcl":
            block, args = TCL, tensor_args
        elif self.block_type == "trl-half":
            tensor_args["core_shape"] = ranks  # ranks if for rnn TODO
            block, args = TRLhalf, tensor_args
        elif self.block_type == "trl3dhalf":
            tensor_args["core_shape"] = ranks
            block, args = TRL3Dhalf, tensor_args
        elif self.block_type == "tt":
            tensor_args["ranks"] = ranks
            block, args = TT, tensor_args
        elif self.block_type == "linear":
            block, args = nn.Linear, {"bias": self.bias_rank}
        elif self.block_type == "identity" or self.block_type == "none":
            block, args = nn.Identity, {}
        elif self.block_type == "flatten":
            block, args = nn.Flatten, {"start_dim": 2}
        elif self.block_type == "linear_norm":
            block, args = LinearNorm, tensor_args
        else:
            raise ValueError(f"Incorrect block type: {self.block_type}.")
        return block, args

    def _sanity_check(self, ranks) -> None:
        # sanity check
        if self.block_type in ["tcl3d", "tcl", "trl-half", "trl3dhalf", "tt"]:
            assert isinstance(
                self.input_dims, tuple
            ), f"Something is wrong with input dims."
            assert isinstance(
                self.hidden_dims, tuple
            ), f"Something is wrong with hidden dims."

        if self.block_type in ["trl-half", "trl3dhalf", "tt"]:
            assert ranks is not None, f"Ranks not provided"

        if self.block_type == "linear":
            if isinstance(self.input_dims, tuple):
                self.input_dims = np.prod(self.input_dims)
                warnings.warn(
                    "Input dims can't be tuple for linear layer, product values."
                )
            if isinstance(self.hidden_dims, tuple):
                self.hidden_dims = np.prod(self.hidden_dims)
                warnings.warn(
                    "Hidden dims can't be tuple for linear layer, product values."
                )
            if self.bias_rank != 0:
                self.bias_rank = 1

        if self.block_place == "output":
            if self.block_type.startswith("tcl") or self.block_type == "tt":
                if self.hidden_dims != (1, 1, 1):
                    warnings.warn(
                        f"Wrong output dims {self.hidden_dims} for layer {self.block_type}."
                    )
                    warnings.warn(f"Set output dims to (1, 1, 1).")
                    self.hidden_dims = (1, 1, 1)
            elif self.block_type.startswith("trl"):
                if self.hidden_dims != (1,):
                    warnings.warn(
                        f"Wrong output dims {self.hidden_dims} for layer {self.block_type}."
                    )
                    warnings.warn(f"Set output dims to (1, ).")
                    self.hidden_dims = (1,)
            elif self.block_type == "linear":
                if self.hidden_dims != 1:
                    warnings.warn(
                        f"Wrong output dims {self.hidden_dims} for layer {self.block_type}."
                    )
                    warnings.warn(f"Set output dims to 1.")
                    self.hidden_dims = 1

    def _process_special_block(self, block) -> nn.Module:
        if self.block_place == "input":
            if self.block_type not in ["flatten", "none", "linear"]:
                _block = nn.Sequential(
                    block(self.input_dims, self.hidden_dims, **self.args), nn.ReLu()
                )
            elif self.block_type == "linear":
                _block = nn.Sequential(
                    nn.Flatten(start_dim=2),
                    block(self.input_dims, self.hidden_dims, **self.args),
                    nn.ReLu(),
                )
            elif self.block_type == "flatten":
                _block = nn.Flatten(**self.args)
            else:
                _block = block(self.input_dims, self.hidden_dims, **self.args)

        elif self.block_place == "output":
            if self.block_type == "linear":
                _block = nn.Sequential(
                    nn.Flatten(start_dim=2),
                    block(self.input_dims, self.hidden_dims, **self.args),
                )
            else:
                _block = block(self.input_dims, self.hidden_dims, **self.args)
        return _block


class GruTl(nn.Module):
    def __init__(self, input_layer, hidden_layer, batch_first=True) -> None:
        super().__init__()

        self.linear_w = nn.ModuleList([input_layer for _ in range(3)])
        self.linear_u = nn.ModuleList([hidden_layer for _ in range(3)])

        self.batch_first = batch_first

    def forward(self, inputs, h_prev=None):
        if self.batch_first:
            # sequence first (timesteps, batch_size, input_dims)
            inputs = inputs.transpose(0, 1)

        outputs = []
        if h_prev is None:
            h_prev = torch.zeros(inputs.shape[1], *self.hidden_dims).to(inputs)

        for x_t in inputs:
            x_z, x_r, x_h = [linear(x_t) for linear in self.linear_w]
            h_z, h_r, h_h = [linear(h_prev) for linear in self.linear_u]

            output_z = torch.sigmoid(x_z + h_z)
            output_r = torch.sigmoid(x_r + h_r)
            hidden_hat = torch.tanh(x_h + output_r * h_h)
            h_prev = output_z * h_prev + (1 - output_z) * hidden_hat

            outputs.append(h_prev)

        outputs = torch.stack(outputs, dim=0)

        if self.batch_first:
            # batch first (batch_size, timesteps, input_dims)
            outputs = outputs.transpose(0, 1)

        return outputs, h_prev


class LstmTl(nn.Module):
    def __init__(self, input_layer, hidden_layer, batch_first=True) -> None:
        super().__init__()

        self.linear_w = nn.ModuleList([input_layer for _ in range(3)])
        self.linear_u = nn.ModuleList([hidden_layer for _ in range(3)])

        self.batch_first = batch_first

    def forward(self, inputs, init_states=None):
        if self.batch_first:
            # sequence first (timesteps, batch_size, input_dims)
            inputs = inputs.transpose(0, 1)

        outputs = []
        if init_states is None:
            h_prev = torch.zeros(inputs.shape[1], *self.hidden_dims).to(inputs)
            c_prev = torch.zeros(inputs.shape[1], *self.hidden_dims).to(inputs)
        else:
            h_prev, c_prev = init_states

        for x_t in inputs:
            x_f, x_i, x_o, x_c_hat = [linear(x_t) for linear in self.linear_w]
            h_f, h_i, h_o, h_c_hat = [linear(h_prev) for linear in self.linear_u]

            f_t = torch.sigmoid(x_f + h_f)
            i_t = torch.sigmoid(x_i + h_i)
            o_t = torch.sigmoid(x_o + h_o)
            c_t_hat = torch.tanh(x_c_hat + h_c_hat)
            c_prev = torch.mul(f_t, c_prev) + torch.mul(i_t, c_t_hat)
            h_prev = torch.mul(o_t, torch.tanh(c_prev))
            outputs.append(h_prev)

        outputs = torch.stack(outputs, dim=0)

        if self.batch_first:
            # batch first (batch_size, timesteps, input_dims)
            outputs = outputs.transpose(0, 1)

        return outputs, (h_prev, c_prev)


class LinearNorm(nn.Module):
    def __init__(self, input_shape, output_shape, bias=1, normalize="both") -> None:
        super().__init__()

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.normalize = normalize
        if self.normalize in ["both", "in"]:
            self.norm_in = nn.LayerNorm(input_shape)
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape)

        self.linear = nn.Linear(input_shape, output_shape, bias=bias)

    def forward(self, x) -> torch.Tensor:
        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)
        y = self.linear(x)
        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)
        return y
