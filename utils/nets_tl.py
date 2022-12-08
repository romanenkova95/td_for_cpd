import torch
import torch.nn as nn

from typing import Dict, Tuple
import string

import numpy as np

class custom_GRU_TL(nn.Module):

    # TODO add num_layers parameter
    def __init__(self, block_type, input_dims, hidden_dims, ranks=None, bias_rank=False, freeze_modes=None):
        super().__init__()
        
        self.hidden_dims = hidden_dims

        if block_type.lower() == "tcl3d":
            block, args = TCL3D, {}
        elif block_type.lower() == "tcl":
            block, args = TCL, {}
        elif block_type.lower() == "trl":
            block, args = TRL, {"core_shape": ranks}
        elif block_type.lower() == "trl-half":
            block, args = TRLhalf, {"core_shape": ranks}
        elif block_type.lower() == "trl3dhalf":
            block, args = TRL3Dhalf, {"core_shape": ranks}
        else:
            raise ValueError(f'Incorrect block type: {block_type}. Should be tcl or trl')

        self.linear_w = nn.ModuleList([
            block(input_dims, hidden_dims, **args, bias_rank=bias_rank, freeze_modes=freeze_modes) 
            for _ in range(3)])
        
        self.linear_u = nn.ModuleList([
            block(hidden_dims, hidden_dims, **args, bias_rank=bias_rank, freeze_modes=freeze_modes) 
            for _ in range(3)])
        
    
    def forward(self, inputs, h_prev=None):

        # inputs shape: L, N, Hin
        L, N, *input_dims = inputs.shape

        outputs = []
        if h_prev is None:
            h_prev = torch.zeros(N, *self.hidden_dims[1:]).to(inputs)
        
        # print(f'h0: {h_prev.shape}')
        
        for x_t in inputs:
            x_z, x_r, x_h = [linear(x_t) for linear in self.linear_w]
            h_z, h_r, h_h = [linear(h_prev) for linear in self.linear_u]
            
            output_z = torch.sigmoid(x_z + h_z)
            output_r = torch.sigmoid(x_r + h_r)
            hidden_hat = torch.tanh(x_h + output_r*h_h)
            h_prev = output_z * h_prev + (1 - output_z) * hidden_hat

            outputs.append(h_prev)#.clone().detach()) # NOTE fail with detach. check it
        
        outputs = torch.stack(outputs, dim=0)
        # print(f'h1: {h_prev.shape}')

        return outputs, h_prev
    
class custom_LSTM_TL(nn.Module):

    # TODO add num_layers parameter
    def __init__(self, block_type, input_dims, hidden_dims, ranks=None, bias_rank=False, freeze_modes=None):
        super().__init__()
        
        self.hidden_dims = hidden_dims

        if block_type.lower() == "tcl3d":
            block, args = TCL3D, {}
        elif block_type.lower() == "tcl":
            block, args = TCL, {}
        elif block_type.lower() == "trl":
            block, args = TRL, {"core_shape": ranks}
        elif block_type.lower() == "trl-half":
            block, args = TRLhalf, {"core_shape": ranks}
        elif block_type.lower() == "trl3dhalf":
            block, args = TRL3Dhalf, {"core_shape": ranks}
        else:
            raise ValueError(f'Incorrect block type: {block_type}. Should be tcl or trl')

        self.linear_w = nn.ModuleList([
            block(input_dims, hidden_dims, **args, bias_rank=bias_rank, freeze_modes=freeze_modes) 
            for _ in range(4)])
        
        self.linear_u = nn.ModuleList([
            block(hidden_dims, hidden_dims, **args, bias_rank=bias_rank, freeze_modes=freeze_modes) 
            for _ in range(4)])
        
    
    def forward(self, inputs, init_states=None):

        # inputs shape: L, N, Hin
        L, N, *input_dims = inputs.shape

        outputs = []
        if init_states is None:
            h_prev = torch.zeros(N, *self.hidden_dims[1:]).to(inputs)
            c_prev = torch.zeros(N, *self.hidden_dims[1:]).to(inputs)
        else:
            h_prev, c_prev = init_states
        
        for x_t in inputs:
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

        return outputs, (h_prev, c_prev)
        
    
class NetD_TL(nn.Module):
    def __init__(self, args, block_type, bias="none") -> None:
        super().__init__()
        self.RNN_hid_dims = args['RNN_hid_dim']
        self.emb_dims = args['emb_dim']
        self.relu = nn.ReLU()
        self.data_dim = args['data_dim']

        fc_bias  = args['bias_rank'] if bias in ["all", "fc"]   else 0
        gru_bias = args['bias_rank'] if bias in ["all", "gru"] else 0

        if block_type.lower() == "tcl3d":
            block, args_in, args_gru, args_out = TCL3D, {}, {}, {}
        elif block_type.lower() == "tcl":
            block, args_in, args_gru, args_out = TCL, {}, {}, {}
        elif block_type.lower() == "trl":
            block = TRL
            args_in  = {"core_shape": args["ranks_input"]}
            args_gru = {"ranks": args["ranks_gru"]}
            args_out = {"core_shape": args["ranks_output"]}
        elif block_type.lower() == "trl-half":
            block = TRLhalf
            args_in  = {"core_shape": args["ranks_input"]}
            args_gru = {"ranks": args["ranks_gru"]}
            args_out = {"core_shape": args["ranks_output"]}
        else:
            raise ValueError(f'Incorrect block type: {block_type}. Should be tcl or trl')

        self.fc1 = block((1, 1,) + self.data_dim, 
                         (1, 1,) + self.emb_dims, 
                         bias_rank=fc_bias, 
                         freeze_modes=[0, 1],
                         **args_in)

        self.rnn_enc_layer = custom_GRU_TL(block_type,
                                           (1,) + self.emb_dims, 
                                           (1,) + self.RNN_hid_dims, 
                                           bias_rank=gru_bias, 
                                           freeze_modes=[0],
                                           **args_gru) 
                                            #, num_layers=args['num_layers'], batch_first=True)
        
        self.rnn_dec_layer = custom_GRU_TL(block_type,
                                           (1,) + self.RNN_hid_dims, 
                                           (1,) + self.emb_dims, 
                                           bias_rank=gru_bias, 
                                           freeze_modes=[0],
                                           **args_gru) 
                                            #, num_layers=args['num_layers'], batch_first=True)

        self.fc2 = block((1, 1,) + self.emb_dims, 
                         (1, 1,) + self.data_dim, 
                         bias_rank=fc_bias, 
                         freeze_modes=[0, 1],
                         **args_out)
        
    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor]:

        if len(X.shape) == 3:
            # X is flatten
            X = X.reshape(*X.shape[:2], *self.data_dim)

        X = self.relu(self.fc1(X))
        X = X.transpose(0, 1) # sequence first (L, N, input_dims)

        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        X_enc = X_enc.transpose(0, 1) # batch first (L, N, input_dims)
        X_dec = X_dec.transpose(0, 1) # batch first (L, N, input_dims)

        X_dec = self.relu(self.fc2(X_dec))
        return X_enc, X_dec
    

class NetG_TL(nn.Module):
    def __init__(self, args, block_type, bias="none") -> None:
            
        super().__init__()
        self.RNN_hid_dims = args['RNN_hid_dim']
        self.emb_dims = args['emb_dim']
        self.relu = nn.ReLU()

        fc_bias  = args['bias_rank'] if bias in ["all", "fc"]   else 0
        gru_bias = args['bias_rank'] if bias in ["all", "gru"] else 0

        if block_type.lower() == "tcl3d":
            block, args_in, args_gru, args_out = TCL3D, {}, {}, {}
        elif block_type.lower() == "tcl":
            block, args_in, args_gru, args_out = TCL, {}, {}, {}
        elif block_type.lower() == "trl":
            block = TRL
            args_in  = {"core_shape": args["ranks_input"]}
            args_gru = {"ranks": args["ranks_gru"]}
            args_out = {"core_shape": args["ranks_output"]}
        elif block_type.lower() == "trl-half":
            block = TRLhalf
            args_in  = {"core_shape": args["ranks_input"]}
            args_gru = {"ranks": args["ranks_gru"]}
            args_out = {"core_shape": args["ranks_output"]}
        else:
            raise ValueError(f'Incorrect block type: {block_type}. Should be tcl or trl')

        self.fc = block((1, 1,) + args['data_dim'], 
                        (1, 1,) + self.emb_dims, 
                        bias_rank=fc_bias, 
                        freeze_modes=[0, 1],
                        **args_in)

        self.rnn_enc_layer = custom_GRU_TL(block_type,
                                           (1,) + self.emb_dims, 
                                           (1,) + self.RNN_hid_dims, 
                                           bias_rank=gru_bias, 
                                           freeze_modes=[0],
                                           **args_gru) 
                                            #, num_layers=args['num_layers'], batch_first=True)
        
        self.rnn_dec_layer = custom_GRU_TL(block_type,
                                           (1,) + self.emb_dims,
                                           (1,) + self.RNN_hid_dims,  
                                           bias_rank=gru_bias, 
                                           freeze_modes=[0],
                                           **args_gru) 
                                            #, num_layers=args['num_layers'], batch_first=True)

        self.fc_layer = block((1, 1,) + self.RNN_hid_dims, 
                              (1, 1,) + args['data_dim'], 
                              bias_rank=fc_bias, 
                              freeze_modes=[0, 1],
                              **args_out)

    
    def forward(self, X_p, X_f, noise) -> torch.Tensor:

        X_p = self.relu(self.fc(X_p))
        X_p = X_p.transpose(0, 1) # sequence first (L, N, input_dims)

        X_f = X_f.transpose(0, 1) # sequence first (L, N, input_dims)
        X_f = self.relu(self.fc(X_f))

        # FIXME continue from here
        _, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        # print(h_t.shape)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)

        Y_f = Y_f.transpose(0, 1) # batch first (L, N, input_dims)
        # output = output.transpose(0, 1) # batch first (L, N, input_dims)
        output = self.fc_layer(Y_f)

        return output

    def shft_right_one(self, X) -> torch.Tensor:
        X_shft = X.clone()
        X_shft[0].data.fill_(0)
        X_shft[1:] = X[:-1]
        return X_shft

############## Blocks ################

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
    
    def __init__(self, input_shape, output_shape, bias_rank=1, freeze_modes=None, normalize="both", method="no_einsum") -> None:
        super().__init__()

        self.freeze_modes = freeze_modes if freeze_modes is not None else []
        # self.order = np.delete(np.arange(len(input_shape)), self.freeze_modes)[::-1]

        self.l_fm = len(self.freeze_modes)
        self.bias_rank = bias_rank
        self.method = method
        self.normalize = normalize
            
        n, m = len(input_shape), len(output_shape)
        l_nfm = n - self.l_fm
        # assert n == m and l_nfm == 3 and np.all(np.array(freeze_modes) < l_nfm), \
        assert n == m and l_nfm == 3 and np.all(np.array(self.freeze_modes) < l_nfm), \
            f'Some shape is incorrect: input {n} != output {m}, skip modes should go first {self.l_fm}'
        
        if method == "einsum":
            self.factors = nn.ParameterList([
                nn.Parameter(torch.randn((size_in, size_out))) 
                for size_in, size_out in zip(input_shape[self.l_fm:], output_shape[self.l_fm:])
                ])
            self.rule = f'...abc,ad,be,cf->...def'
            # print(f'TCL: {self.rule}', input_shape, output_shape)
        else:
            self.factors = nn.ModuleList([
            nn.Linear(size_in, size_out, bias=False) 
            for size_in, size_out in zip(input_shape[self.l_fm:], output_shape[self.l_fm:])
            ])

        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)

        if self.normalize in ["both", "in"]:
            self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])

    def forward(self, x) -> torch.Tensor:

        # print(f'TCL forward: {self.rule}, X shape: {x.shape}')
        # print(f'x: {x.device}, w: {self.factors[0].weight.device}, b {self.bias.device}')

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
    
    def __init__(self, input_shape, output_shape, bias_rank=1, freeze_modes=None, normalize="both") -> None:
        super().__init__()

        self.normalize = normalize
        self.bias_rank = bias_rank

        if freeze_modes is None:
            freeze_modes = []

        self.l_fm = len(freeze_modes)
            
        n, m = len(input_shape), len(output_shape)
        assert n == m and n < 12, \
            f'Some shape is incorrect: input {n} != output {m}'
        
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn((size_in, size_out))) 
            for s, (size_in, size_out) in enumerate(zip(input_shape, output_shape))
            if s not in freeze_modes
            ])

        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)

        chars_inner = string.ascii_lowercase[:n]
        chars_outer = string.ascii_lowercase[n:2*n]
        chars_middle, chars_outer_new = [], []
        for s, (i, j) in enumerate(zip(chars_inner, chars_outer)):
            if s not in freeze_modes:
                chars_middle += [i+j]
                chars_outer_new += [j]
            else:
                chars_outer_new += [i]
                
        self.rule = f'{chars_inner},' \
                    f'{",".join(chars_middle)}' \
                    f'->{"".join(chars_outer_new)}'
        # print(f'TCL: {self.rule}', input_shape, output_shape)

        if self.normalize in ["both", "in"]:
            self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])


    def forward(self, x) -> torch.Tensor:

        # print(f'TCL forward: {self.rule}, X shape: {x.shape}')
        # print(f'x: {x.device}, w: {self.factors[0].device}')
        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        y = torch.einsum(self.rule, x, *self.factors)
        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y


class TRL(nn.Module):

    def __init__(self, input_shape, output_shape, core_shape, bias_rank=1, freeze_modes=None, normalize="both") -> None:
        super().__init__()
        
        self.normalize = normalize
        self.bias_rank = bias_rank

        if freeze_modes is None:
            freeze_modes = []

        self.l_fm = len(freeze_modes)
        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        l_nfm = n - self.l_fm
        assert n + m == l + 2 * self.l_fm and l < 12, f'' \
            f'Some shape is incorrect: input {n} + output {m} != core {l} + 2 freeze_modes {self.l_fm}'
        
        input_shape_filtered  = [size for s, size in enumerate(input_shape)  if s not in freeze_modes]
        output_shape_filtered = [size for s, size in enumerate(output_shape) if s not in freeze_modes]

        self.factors_inner = nn.ParameterList([
            nn.Parameter(torch.randn((size_in, size_out))) 
            for s, (size_in, size_out) in enumerate(zip(input_shape_filtered, core_shape[:l_nfm]))
            ])
        self.factors_outer = nn.ParameterList([
            nn.Parameter(torch.randn((size_in, size_out))) 
            for s, (size_in, size_out) in enumerate(zip(core_shape[l_nfm:], output_shape_filtered))
            ])
        self.core = nn.Parameter(torch.randn(core_shape))
        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)
        
        chars_inner = string.ascii_lowercase[:n]
        chars_core  = string.ascii_lowercase[n:2*n]
        chars_outer = string.ascii_lowercase[2*n:2*n+m]
        chars_final = string.ascii_lowercase[2*n+m:2*n+2*m]

        chars_middle1, chars_middle2, buffer = [], [], {}
        chars_core_new, chars_outer_new, chars_final_new = [], [], []

        for s, (i, j) in enumerate(zip(chars_inner, chars_core)):
            if s not in freeze_modes:
                chars_middle1 += [i+j]
                chars_core_new += [j]
            else:
                buffer[s] = i

        for s, (i, j) in enumerate(zip(chars_outer, chars_final)):
            if s not in freeze_modes:
                chars_middle2 += [i+j]
                chars_outer_new += [i]
                chars_final_new += [j]
            else:
                chars_final_new += [buffer[s]]

        self.rule = f'{chars_inner},' \
            f'{",".join(chars_middle1)}' \
            f',{"".join(chars_core_new)}{"".join(chars_outer_new)},' \
            f'{",".join(chars_middle2)}' \
            f'->{"".join(chars_final_new)}'

        # print(f'TRL: {self.rule}')

        if self.normalize in ["both", "in"]:
            self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        if self.normalize in ["both", "out"]:
            self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])



    def forward(self, x) -> torch.Tensor:

        if self.normalize in ["both", "in"]:
            x = self.norm_in(x)

        # print(f'TRL forward: {self.rule}, X shape: {x.shape}, {len(self.factors_inner)}, {len(self.factors_outer)}')
        y = torch.einsum(self.rule, x, *self.factors_inner, self.core, *self.factors_outer)
        y += add_bias_Nd(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y


class TRLhalf(nn.Module):

    def __init__(self, input_shape, output_shape, core_shape, bias_rank=0, freeze_modes=None, normalize="both") -> None:
        super().__init__()
        
        self.normalize = normalize
        self.bias_rank = bias_rank

        if freeze_modes is None:
            freeze_modes = []

        self.l_fm = len(freeze_modes)
        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        l_nfm = n - self.l_fm
        assert l_nfm == l and l < 12, f'' \
            f'Some shape is incorrect: input {n} + output {m} != core {l} + 2 freeze_modes {self.l_fm}'

        self.factors_inner = nn.ParameterList([
            nn.Parameter(torch.randn((size_in, size_out))) 
            for size_in, size_out in zip(input_shape[self.l_fm:], core_shape)
            ])
        self.core = nn.Parameter(torch.randn(*core_shape, *output_shape[self.l_fm:]))
        self.bias, self.bias_list = initialize_bias(output_shape[self.l_fm:], bias_rank)
        
        chars_inner = string.ascii_lowercase[:n]
        chars_core  = string.ascii_lowercase[n:2*n]
        chars_final = string.ascii_lowercase[2*n:2*n+m-self.l_fm]

        chars_middle, chars_frozen, chars_core_new = [], [], []

        for s, (i, j) in enumerate(zip(chars_inner, chars_core)):
            if s not in freeze_modes:
                chars_middle += [i+j]
                chars_core_new += [j]
            else:
                chars_frozen += [i]

        self.rule = f'{chars_inner},' \
            f'{",".join(chars_middle)}' \
            f',{"".join(chars_core_new)}{chars_final}' \
            f'->{"".join(chars_frozen)}{chars_final}'

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

        y += add_bias_3d(self.bias, self.bias_rank, self.bias_list)

        if self.normalize in ["both", "out"]:
            y = self.norm_out(y)

        return y

    
#############################BCE MODEL##################################
class BCE_GRU_TL(nn.Module):
    def __init__(self, args, block_type, bias="none") -> None:
        super().__init__()
        
        self.RNN_hid_dims = args['RNN_hid_dim']
        self.emb_dims = args['emb_dim']
        self.relu = nn.ReLU()
        self.data_dim = args['data_dim']

        fc_bias  = args['bias_rank'] if bias in ["all", "fc"]   else 0
        gru_bias = args['bias_rank'] if bias in ["all", "gru"] else 0

        if block_type.lower() == "tcl3d":
            block, args_in, args_gru, args_out = TCL3D, {}, {}, {}
        elif block_type.lower() == "tcl":
            block, args_in, args_gru, args_out = TCL, {}, {}, {}
        elif block_type.lower() == "trl":
            block = TRL
            args_in  = {"core_shape": args["ranks_input"]}
            args_gru = {"ranks": args["ranks_gru"]}
            args_out = {"core_shape": args["ranks_output"]}
        elif block_type.lower() == "trl-half":
            block = TRLhalf
            args_in  = {"core_shape": args["ranks_input"]}
            args_gru = {"ranks": args["ranks_gru"]}
            args_out = {"core_shape": args["ranks_output"]}
        else:
            raise ValueError(f'Incorrect block type: {block_type}. Should be tcl or trl')
             

        #self.fc_1 = block((1, 1,) + self.data_dim, 
        #                  (1, 1,) + self.emb_dims, 
        #                  bias_rank=fc_bias, 
        #                  freeze_modes=[0, 1],
        #                  **args_in)
                
        self.rnn = custom_GRU_TL(block_type,
                                 (1,) + self.data_dim, 
                                 (1,) + self.RNN_hid_dims, 
                                 bias_rank=gru_bias, 
                                 freeze_modes=[0],
                                 **args_gru) 
        
        self.fc_2 = block((1, 1,) + self.RNN_hid_dims, 
                          (1, 1,) + (1, 1, 1), 
                          bias_rank=fc_bias, 
                          freeze_modes=[0, 1],
                          normalize="in",
                          **args_out)
        
    def forward(self, x):
        #x = self.relu(self.fc_1(x)) # batch_size, timesteps, C, H, W
        x = x.transpose(0, 1) # sequence first (timesteps, batch_size, input_dims)
        x, _ = self.rnn(x)
        x = x.transpose(0, 1) # batch first (batch_size, timesteps, input_dims)
        x = self.fc_2(x) 
        
        x = x.reshape(*x.shape[:2], 1)# flatten accross last 3 dim
        x = torch.sigmoid(x)
        return x


def parse_bce_args(args: Dict):

        block_type = args["block_type"].lower()
        rnn_type = args["rnn_type"].lower()

        fc_bias  = args['bias_rank'] if args["bias"] in ["all", "fc" ] else 0
        gru_bias = args['bias_rank'] if args["bias"] in ["all", "gru"] else 0

        block_in = None
        if block_type in ["linear"]:
            block_rnn = nn.LSTM if rnn_type in ["lstm"] else nn.GRU
            flatten_trl = args["flatten_type"] == "trl"
            input_dim = args['emb_dim'] if flatten_trl else args['data_dim']
            args_rnn = {
                "input_size": input_dim, 
                "hidden_size": args['RNN_hid_dim'], 
                "num_layers": args['num_layers'], 
                "batch_first": True,
                "bias": gru_bias > 0
                }
            block = nn.Linear
            args_fc = {
                "in_features": args['RNN_hid_dim'], 
                "out_features": 1, 
                "bias": fc_bias > 0
                }
            # if flatten_trl:
            #     block_in = TRL3Dhalf(
            #         input_shape=(1, 1,) + args['data_dim'], 
            #         output_shape=(1, 1, args['emb_dim']),
            #         core_shape=args['data_dim'],
            #         bias_rank=fc_bias, 
            #         freeze_modes=[0, 1],
            #         normalize="both")
        else:
            block_rnn = custom_LSTM_TL if rnn_type in ["lstm"] else custom_GRU_TL
            args_rnn = {
                "block_type": block_type, 
                "input_dims": (1,) + args['data_dim'], 
                "hidden_dims": (1,)+ args['RNN_hid_dim'], 
                "bias_rank": gru_bias, 
                "freeze_modes": [0]
                }
            args_fc = {
                "input_shape": (1, 1,) + args['RNN_hid_dim'], 
                "output_shape": (1, 1, 1, 1, 1),
                "bias_rank": fc_bias, 
                "freeze_modes": [0, 1],
                "normalize": "in"
                }

            if block_type == "tcl3d":
                block = TCL3D
            elif block_type == "tcl":
                block = TCL
            elif block_type == "trl":
                block = TRL
            elif block_type == "trl-half":
                block = TRLhalf
            elif block_type == "trl3dhalf":
                block = TRL3Dhalf
            else:
                raise ValueError(f'Incorrect block type: {block_type}. Should be tcl or trl')

        if block_type in ["trl", "trl-half", "trl3dhalf"]:
            args_fc["core_shape"] = args["ranks_input"]
            args_rnn["ranks"] = args["ranks_rnn"]

        is_tl = block_type not in ["linear", "masked"]

        return block_rnn, block, args_rnn, args_fc, block_in, is_tl


class BCE_GRU_TL_v2(nn.Module):
    def __init__(self, args: Dict) -> None:
        super().__init__()
        
        self.relu = nn.ReLU()
        block_rnn, block, args_rnn, args_fc, args_in, self.is_tl = parse_bce_args(args)

        self.rnn = block_rnn(**args_rnn) 
        self.fc_2 = block(**args_fc)
        
    def forward_tl(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.relu(self.fc_1(x)) # batch_size, timesteps, C, H, W
        x = x.transpose(0, 1) # sequence first (timesteps, batch_size, input_dims)
        x, _ = self.rnn(x)
        x = x.transpose(0, 1) # batch first (batch_size, timesteps, input_dims)
        x = self.fc_2(x) 
        
        x = x.reshape(*x.shape[:2], 1)# flatten accross last 3 dim
        x = torch.sigmoid(x)
        return x
            
    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2)
        x, _ = self.rnn(x)
        x = self.fc_2(x) # batch_size, timesteps, 1  
        #x = x.reshape(*x.shape[:2], 1)
        x = torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tl(x) if self.is_tl else self.forward_linear(x)
