# Tensor Layers and help functions
import torch
import torch.nn as nn

from typing import Dict, Tuple
import string

import numpy as np


def initialize_bias(output_shape, bias_rank):
    if bias_rank == -1:
        bias_list = []
        bias = nn.Parameter(torch.randn(output_shape))
    elif bias_rank > 0:
        bias_list = nn.ParameterList(
            [
                nn.Parameter(torch.randn(bias_rank, size_out))
                for size_out in output_shape
            ]
        )
        bias = None
    else:
        bias_list = []
        bias = None

    return bias, bias_list


def forward_tcl_3d(x, factors, feature_dims=3):
    nfm = x.ndim - feature_dims
    y = factors[2](x)
    y = factors[1](y.transpose(-2, -1))
    y = factors[0](y.transpose(-3, -1))
    y = y.permute(*np.arange(nfm), nfm + 2, nfm + 0, nfm + 1)

    return y


def forward_tcl_Nd(x, factors, feature_dims):
    N = x.ndim
    y = x
    for i in range(feature_dims):
        y = factors[feature_dims - i - 1](y.transpose(N - i - 1, -1))

    nfm = N - feature_dims
    arange_fm = np.arange(nfm, N)
    y = y.permute(*np.arange(nfm), arange_fm[-1], *arange_fm[:-1])

    return y


def add_bias_3d(bias, bias_rank, bias_list):
    y = 0
    if bias is not None:
        y = bias
    elif bias_rank > 0:
        for i in range(bias_rank):
            y += (
                bias_list[0][i][:, None, None]
                * bias_list[1][i][None, :, None]
                * bias_list[2][i][None, None, :]
            )
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


# TODO: replace freeze_modes with feature dims
class TCL3D(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        bias_rank=1,
        normalize="both",
        method="no_einsum",
    ) -> None:
        super().__init__()

        self.fdim = len(input_shape)
        self.bias_rank = bias_rank
        self.method = method
        self.normalize = normalize

        n, m = len(input_shape), len(output_shape)
        assert n == m, f"Some shape is incorrect: input {n} != output {m}"

        if method == "einsum":
            self.factors = nn.ParameterList(
                [
                    nn.Parameter(torch.randn((size_in, size_out)))
                    for size_in, size_out in zip(input_shape, output_shape)
                ]
            )
            self.rule = f"...abc,ad,be,cf->...def"
        else:
            self.factors = nn.ModuleList(
                [
                    nn.Linear(size_in, size_out, bias=False)
                    for size_in, size_out in zip(input_shape, output_shape)
                ]
            )

        self.bias, self.bias_list = initialize_bias(output_shape, bias_rank)

        if self.normalize in ["both", "in"]:
            self.norm_in = nn.LayerNorm(input_shape)
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape)

    def forward(self, x) -> torch.Tensor:
        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        if self.method == "einsum":
            y = torch.einsum(self.rule, x, *self.factors)
        else:
            y = forward_tcl_3d(x, self.factors, self.fdim)

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
        normalize="both",
        method="no_einsum",
    ) -> None:
        super().__init__()

        self.fdim = len(input_shape)
        self.normalize = normalize
        self.bias_rank = bias_rank
        self.method = method

        n, m = len(input_shape), len(output_shape)
        assert n == m, f"Some shape is incorrect: input {n} != output {m}"

        if method == "einsum":
            self.factors = nn.ParameterList(
                [
                    nn.Parameter(torch.randn((size_in, size_out)))
                    for size_in, size_out in zip(input_shape, output_shape)
                ]
            )
        else:
            self.factors = nn.ModuleList(
                [
                    nn.Linear(size_in, size_out, bias=False)
                    for size_in, size_out in zip(input_shape, output_shape)
                ]
            )

        self.bias, self.bias_list = initialize_bias(output_shape, bias_rank)

        chars_inner = string.ascii_lowercase[:n]
        chars_outer = string.ascii_lowercase[n : 2 * n]
        chars_middle = [i + j for i, j in zip(chars_inner, chars_outer)]

        self.rule = (
            f"...{chars_inner},"
            f'{",".join(chars_middle)}'
            f'->...{"".join(chars_outer)}'
        )
        print(f"TCL: {self.rule}", input_shape, output_shape)

        if self.normalize in ["both", "in"]:
            self.norm_in = nn.LayerNorm(input_shape)
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape)

    def forward(self, x) -> torch.Tensor:
        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        if self.method == "einsum":
            y = torch.einsum(self.rule, x, *self.factors)
        else:
            y = forward_tcl_Nd(x, self.factors, self.fdim)

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
        bias_rank=1,
        normalize="both",
        method="no_einsum",
    ) -> None:
        super().__init__()

        self.normalize = normalize
        self.bias_rank = bias_rank
        self.method = method
        self.fdim = len(input_shape)
        self.core_in_features = np.prod(core_shape)
        self.core_out_features = output_shape

        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        assert n == l, f"Some shape is incorrect: input {n} + output{m} != core {l}"

        if method == "einsum":
            self.factors = nn.ParameterList(
                [
                    nn.Parameter(torch.randn((size_in, size_out)))
                    for size_in, size_out in zip(input_shape, core_shape)
                ]
            )

            self.core = nn.Parameter(torch.randn(*core_shape, *output_shape))

            chars_inner = string.ascii_lowercase[:n]
            chars_core = string.ascii_lowercase[n : 2 * n]
            chars_final = string.ascii_lowercase[2 * n : 2 * n + m]
            chars_middle = [i + j for i, j in zip(chars_inner, chars_core)]

            self.rule = (
                f"...{chars_inner},"
                f'{",".join(chars_middle)}'
                f',{"".join(chars_core)}{chars_final}'
                f"->...{chars_final}"
            )

            print(f"TRL: {self.rule}")
        else:
            self.factors = nn.ModuleList(
                [
                    nn.Linear(size_in, size_out, bias=False)
                    for size_in, size_out in zip(input_shape, core_shape)
                ]
            )
            self.core = nn.Linear(
                self.core_in_features, np.prod(self.core_out_features), bias=False
            )

        self.bias, self.bias_list = initialize_bias(output_shape, bias_rank)
        if self.normalize in ["both", "in"]:
            self.norm_in = nn.LayerNorm(input_shape)
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape)

    def forward(self, x) -> torch.Tensor:
        freeze_shape = tuple(x.shape[: -self.fdim])
        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        if self.method == "einsum":
            # print(f'TRL forward: {self.rule}, X shape: {x.shape}, {len(self.factors_inner)}, {self.core.shape}')
            y = torch.einsum(self.rule, x, *self.factors, self.core)
        else:
            y = forward_tcl_Nd(x, self.factors, self.fdim)
            y = y.reshape(*freeze_shape, self.core_in_features)
            y = self.core(y)
            y = y.reshape(*freeze_shape, *self.core_out_features)

        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y


class TRL3Dhalf(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        core_shape,
        bias_rank=1,
        normalize="both",
        method="no_einsum",
    ) -> None:
        super().__init__()

        self.bias_rank = bias_rank
        self.method = method
        self.normalize = normalize
        self.core_in_features = np.prod(core_shape)
        self.core_out_features = output_shape
        self.fdim = len(input_shape)

        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        assert n == l, f"Some shape is incorrect: input {n} + output {m} != core {l}"

        if method == "einsum":
            self.factors = nn.ParameterList(
                [
                    nn.Parameter(torch.randn((size_in, size_out)))
                    for size_in, size_out in zip(input_shape, core_shape)
                ]
            )
            self.core = nn.Parameter(torch.randn(*core_shape, *output_shape))
            self.rule = (
                f"...abc,ad,be,cf,defghk->...ghk"  # FIXME not correct for input block
            )
            print(f"TRL3D: {self.rule}", input_shape, output_shape)
        else:
            self.factors = nn.ModuleList(
                [
                    nn.Linear(size_in, size_out, bias=False)
                    for size_in, size_out in zip(input_shape, core_shape)
                ]
            )
            self.core = nn.Linear(
                self.core_in_features, np.prod(self.core_out_features), bias=False
            )

        self.bias, self.bias_list = initialize_bias(output_shape, bias_rank)

        if self.normalize in ["both", "in"]:
            self.norm_in = nn.LayerNorm(input_shape)
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape)

    def forward(self, x) -> torch.Tensor:
        freeze_shape = tuple(x.shape[: -self.fdim])
        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        if self.method == "einsum":
            print(f"TRL3Dhalf forward: {self.rule}, X shape: {x.shape}")
            y = torch.einsum(self.rule, x, *self.factors, self.core)
        else:
            y = forward_tcl_3d(x, self.factors, self.fdim)
            y = y.reshape(*freeze_shape, self.core_in_features)
            y = self.core(y)
            y = y.reshape(*freeze_shape, *self.core_out_features)

        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y


def forward_tt_Nd(input, factors, tt_ranks):
    N = input.ndim
    fdim = len(tt_ranks) - 1
    y = torch.unsqueeze(input, -1)

    for i in range(fdim):
        y = y.transpose(N - i - 1, -2)
        shape_local = y.shape[:-2]
        y = y.reshape(*shape_local, -1)
        y = factors[fdim - i - 1](y)
        y = y.reshape(*shape_local, -1, tt_ranks[fdim - i - 1])

    y = torch.squeeze(y, -1)

    nfm = N - fdim
    arange_fm = np.arange(nfm, N)
    y = y.permute(*np.arange(nfm), arange_fm[-1], *arange_fm[:-1])

    return y


class TT(nn.Module):
    def __init__(
        self, input_shape, output_shape, ranks, bias_rank=1, normalize="both"
    ) -> None:
        super().__init__()

        self.bias_rank = bias_rank
        self.normalize = normalize
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.ranks = (1,) + tuple(ranks) + (1,)
        self.fdim = len(input_shape)

        n, m, r = len(input_shape), len(output_shape), len(ranks)
        assert (
            n == m == r + 1
        ), f"Some shape is incorrect: input {n} != output {m} != rank {r} + 1"

        ranks_expanded = (1,) + ranks + ()
        self.factors = nn.ModuleList(
            [
                nn.Linear(
                    self.input_shape[i] * self.ranks[i + 1],
                    self.output_shape[i] * self.ranks[i],
                    bias=False,
                )
                for i in range(self.fdim)
            ]
        )

        self.bias, self.bias_list = initialize_bias(output_shape, bias_rank)

        if self.normalize in ["both", "in"]:
            self.norm_in = nn.LayerNorm(input_shape)
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape)

    def forward(self, x) -> torch.Tensor:
        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        y = forward_tt_Nd(x, self.factors, self.ranks)

        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y
