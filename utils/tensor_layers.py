# Tensor Layers and help functions
import torch
import torch.nn as nn

from typing import Dict, Tuple
import string

import numpy as np


def initialize_bias(output_shape, bias_rank):

    if bias_rank == "full":
        bias_list = []
        bias = nn.Parameter(torch.randn(output_shape))
    elif bias_rank > 0:
        bias_list = nn.ParameterList([
            nn.Parameter(torch.randn(bias_rank, size_out))
            for size_out in output_shape
        ])
        bias = None
    else:
        bias_list = []
        bias = None

    return bias, bias_list


def forward_tcl_3d(x, factors, number_frozen_modes):

    nfm = number_frozen_modes

    y = factors[2](x)
    y = factors[1](y.transpose(-2, -1))
    y = factors[0](y.transpose(-3, -1))
    y = y.permute(*np.arange(nfm), nfm+2, nfm+0, nfm+1)

    return y


def add_bias_3d(bias, bias_rank, bias_list):
    y = 0
    if bias is not None:
        y = bias
    elif bias_rank > 0:
        for i in range(bias_rank):
            y += bias_list[0][i][:, None, None] \
                * bias_list[1][i][None, :, None] \
                * bias_list[2][i][None, None, :]
    return y


def add_bias_Nd(bias, bias_rank, bias_list):
    y = 0
    if bias is not None:
        y = bias
    elif bias_rank > 0:
        for i in range(bias_rank):
            bias = 1
            shape = [1] * len(bias_list)
            for j, bias_vector in enumerate(bias_list):
                shape[j] = -1
                bias = bias * bias_vector[i].reshape(tuple(shape))
                shape[j] = 1

            y += bias

    return y


class TCL3D(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        bias_rank=1,
        freeze_modes=None,
        normalize="both",
        method="no_einsum",
    ) -> None:
        super().__init__()

        self.freeze_modes = freeze_modes if freeze_modes is not None else []

        self.l_fm = len(self.freeze_modes)
        self.bias_rank = bias_rank
        self.method = method
        self.normalize = normalize

        n, m = len(input_shape), len(output_shape)
        l_nfm = n - self.l_fm
        assert (
            n == m and l_nfm == 3 and np.all(np.array(self.freeze_modes) < l_nfm)
        ), f"Some shape is incorrect: input {n} != output {m}, skip modes should go first {self.l_fm}"

        if method == "einsum":
            self.factors = nn.ParameterList(
                [
                    nn.Parameter(torch.randn((size_in, size_out)))
                    for size_in, size_out in zip(
                        input_shape[self.l_fm :], output_shape[self.l_fm :]
                    )
                ]
            )
            self.rule = f"...abc,ad,be,cf->...def"
        else:
            self.factors = nn.ModuleList(
                [
                    nn.Linear(size_in, size_out, bias=False)
                    for size_in, size_out in zip(
                        input_shape[self.l_fm :], output_shape[self.l_fm :]
                    )
                ]
            )

        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)

        if self.normalize in ["both", "in"]:
            self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])

    def forward(self, x) -> torch.Tensor:

        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        if self.method == "einsum":
            y = torch.einsum(self.rule, x, *self.factors)
        else:
            y = forward_tcl_3d(x, self.factors, self.l_fm)

        y += add_bias_3d(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y


class TCL(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        bias_rank=1,
        freeze_modes=None,
        normalize="both",
    ) -> None:
        super().__init__()

        self.normalize = normalize
        self.bias_rank = bias_rank

        if freeze_modes is None:
            freeze_modes = []

        self.l_fm = len(freeze_modes)

        n, m = len(input_shape), len(output_shape)
        assert n == m and n < 12, f"Some shape is incorrect: input {n} != output {m}"

        self.factors = nn.ParameterList(
            [
                nn.Parameter(torch.randn((size_in, size_out)))
                for s, (size_in, size_out) in enumerate(zip(input_shape, output_shape))
                if s not in freeze_modes
            ]
        )

        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)

        chars_inner = string.ascii_lowercase[:n]
        chars_outer = string.ascii_lowercase[n : 2 * n]
        chars_middle, chars_outer_new = [], []
        for s, (i, j) in enumerate(zip(chars_inner, chars_outer)):
            if s not in freeze_modes:
                chars_middle += [i + j]
                chars_outer_new += [j]
            else:
                chars_outer_new += [i]

        self.rule = (
            f"{chars_inner},"
            f'{",".join(chars_middle)}'
            f'->{"".join(chars_outer_new)}'
        )
        # print(f'TCL: {self.rule}', input_shape, output_shape)

        if self.normalize in ["both", "in"]:
            self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])


    def forward(self, x) -> torch.Tensor:

        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)
        y = torch.einsum(self.rule, x, *self.factors)
        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y


class TRL(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        core_shape,
        bias_rank=1,
        freeze_modes=None,
        normalize="both",
    ) -> None:
        super().__init__()

        self.normalize = normalize
        self.bias_rank = bias_rank

        if freeze_modes is None:
            freeze_modes = []

        self.l_fm = len(freeze_modes)
        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        l_nfm = n - self.l_fm
        assert n + m == l + 2 * self.l_fm and l < 12, (
            f""
            f"Some shape is incorrect: input {n} + output{m} != core {l} + 2 freeze_modes {self.l_fm}"
        )

        input_shape_filtered = [
            size for s, size in enumerate(input_shape) if s not in freeze_modes
        ]
        output_shape_filtered = [
            size for s, size in enumerate(output_shape) if s not in freeze_modes
        ]

        self.factors_inner = nn.ParameterList(
            [
                nn.Parameter(torch.randn((size_in, size_out)))
                for s, (size_in, size_out) in enumerate(
                    zip(input_shape_filtered, core_shape[:l_nfm])
                )
            ]
        )

        self.factors_outer = nn.ParameterList(
            [
                nn.Parameter(torch.randn((size_in, size_out)))
                for s, (size_in, size_out) in enumerate(
                    zip(core_shape[l_nfm:], output_shape_filtered)
                )
            ]
        )

        self.core = nn.Parameter(torch.randn(core_shape))
        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)

        chars_inner = string.ascii_lowercase[:n]
        chars_core = string.ascii_lowercase[n : 2 * n]
        chars_outer = string.ascii_lowercase[2 * n : 2 * n + m]
        chars_final = string.ascii_lowercase[2 * n + m : 2 * n + 2 * m]

        chars_middle1, chars_middle2, buffer = [], [], {}
        chars_core_new, chars_outer_new, chars_final_new = [], [], []

        for s, (i, j) in enumerate(zip(chars_inner, chars_core)):
            if s not in freeze_modes:
                chars_middle1 += [i + j]
                chars_core_new += [j]
            else:
                buffer[s] = i

        for s, (i, j) in enumerate(zip(chars_outer, chars_final)):
            if s not in freeze_modes:
                chars_middle2 += [i + j]
                chars_outer_new += [i]
                chars_final_new += [j]
            else:
                chars_final_new += [buffer[s]]

        self.rule = (
            f"{chars_inner},"
            f'{",".join(chars_middle1)}'
            f',{"".join(chars_core_new)}{"".join(chars_outer_new)},'
            f'{",".join(chars_middle2)}'
            f'->{"".join(chars_final_new)}'
        )

        # print(f'TRL: {self.rule}')

        if self.normalize in ["both", "in"]:
            self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])



    def forward(self, x) -> torch.Tensor:

        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        # print(f'TRL forward: {self.rule}, X shape: {x.shape}, {len(self.factors_inner)}, {len(self.factors_outer)}')
        y = torch.einsum(
            self.rule, x, *self.factors_inner, self.core, *self.factors_outer
        )
        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y


class TRLhalf(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        core_shape,
        bias_rank=0,
        freeze_modes=None,
        normalize="both",
    ) -> None:
        super().__init__()

        self.normalize = normalize
        self.bias_rank = bias_rank

        if freeze_modes is None:
            freeze_modes = []

        self.l_fm = len(freeze_modes)
        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        l_nfm = n - self.l_fm
        assert l_nfm == l and l < 12, (
            f""
            f"Some shape is incorrect: input {n} + output{m} != core {l} + 2 freeze_modes {self.l_fm}"
        )

        self.factors_inner = nn.ParameterList(
            [
                nn.Parameter(torch.randn((size_in, size_out)))
                for size_in, size_out in zip(input_shape[self.l_fm :], core_shape)
            ]
        )

        self.core = nn.Parameter(torch.randn(*core_shape, *output_shape[self.l_fm :]))
        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)

        chars_inner = string.ascii_lowercase[:n]
        chars_core = string.ascii_lowercase[n : 2 * n]
        chars_final = string.ascii_lowercase[2 * n : 2 * n + m - self.l_fm]

        chars_middle, chars_frozen, chars_core_new = [], [], []

        for s, (i, j) in enumerate(zip(chars_inner, chars_core)):
            if s not in freeze_modes:
                chars_middle += [i + j]
                chars_core_new += [j]
            else:
                chars_frozen += [i]

        self.rule = (
            f"{chars_inner},"
            f'{",".join(chars_middle)}'
            f',{"".join(chars_core_new)}{chars_final}'
            f'->{"".join(chars_frozen)}{chars_final}'
        )

        # print(f'TRL: {self.rule}')

        if self.normalize in ["both", "in"]:
            self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])

    def forward(self, x) -> torch.Tensor:

        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        # print(f'TRL forward: {self.rule}, X shape: {x.shape}, {len(self.factors_inner)}, {self.core.shape}')
        y = torch.einsum(self.rule, x, *self.factors_inner, self.core)
        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y


class TRL3Dhalf(nn.Module):

    def __init__(self, input_shape, output_shape, core_shape, bias_rank=1, freeze_modes=None, normalize="both", method="no_einsum") -> None:
        super().__init__()

        self.freeze_modes = freeze_modes if freeze_modes is not None else []
        # self.order = np.delete(np.arange(len(input_shape)), self.freeze_modes)[::-1]

        self.l_fm = len(self.freeze_modes)
        self.bias_rank = bias_rank
        self.method = method
        self.normalize = normalize
        self.core_in_features = np.prod(core_shape)
        self.core_out_features = tuple(output_shape[self.l_fm:])

        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        l_nfm = n - self.l_fm
        assert l_nfm == l and l < 12, f'' \
            f'Some shape is incorrect: input {n} + output {m} != core {l} + 2 freeze_modes {self.l_fm}'

        if method == "einsum":
            self.factors = nn.ParameterList([
                nn.Parameter(torch.randn((size_in, size_out)))
                for size_in, size_out in zip(input_shape[self.l_fm:], core_shape)
            ])
            self.core = nn.Parameter(torch.randn(*core_shape, *output_shape[self.l_fm:]))
            self.rule = f'...abc,ad,be,cf,defghk->...ghk'
            print(f'TCL: {self.rule}', input_shape, output_shape)
        else:
            self.factors = nn.ModuleList([
                nn.Linear(size_in, size_out, bias=False)
                for size_in, size_out in zip(input_shape[self.l_fm:], core_shape)
            ])
            self.core = nn.Linear(self.core_in_features, np.prod(self.core_out_features), bias=False)

        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)

        if self.normalize in ["both", "in"]:
            self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])

    def forward(self, x) -> torch.Tensor:

        freeze_shape = tuple(x.shape[:self.l_fm])
        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        if self.method == "einsum":
            print(f'TRL3Dhalf forward: {self.rule}, X shape: {x.shape}')
            y = torch.einsum(self.rule, x, *self.factors, self.core)
        else:
            y = forward_tcl_3d(x, self.factors, self.l_fm)
            y = y.reshape(*freeze_shape, self.core_in_features)
            y = self.core(y)
            y = y.reshape(*freeze_shape, *self.core_out_features)

        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y