import torch
import torch.nn as nn

from typing import Tuple
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
    
class NetD_TL(nn.Module):
    def __init__(self, args, block_type, bias="none") -> None:
        super().__init__()
        self.RNN_hid_dims = args['RNN_hid_dim']
        self.emb_dims = args['emb_dim']
        self.relu = nn.ReLU()
        self.data_dim = args['data_dim']

        fc_bias  = bias in ["all", "fc"] 
        gru_bias = bias in ["all", "gru"]

        if block_type.lower() == "tcl3d":
            block, args_in, args_gru, args_out = TCL, {}, {}, {}
            fc_bias  = args['bias_rank'] if fc_bias  else 0
            gru_bias = args['bias_rank'] if gru_bias else 0
        elif block_type.lower() == "tcl":
            block, args_in, args_gru, args_out = TCL, {}, {}, {}
        elif block_type.lower() == "trl":
            block = TRL
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

        fc_bias  = bias in ["all", "fc"] 
        gru_bias = bias in ["all", "gru"]

        if block_type.lower() == "tcl3d":
            block, args_in, args_gru, args_out = TCL, {}, {}, {}
            fc_bias  = args['bias_rank'] if fc_bias  else 0
            gru_bias = args['bias_rank'] if gru_bias else 0
        elif block_type.lower() == "tcl":
            block, args_in, args_gru, args_out = TCL, {}, {}, {}
        elif block_type.lower() == "trl":
            block = TRL
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


class TCL3D(nn.Module):
    
    def __init__(self, input_shape, output_shape, bias_rank=1, freeze_modes=None, method="einsum") -> None:
        super().__init__()

        self.freeze_modes = freeze_modes if freeze_modes is not None else []
        # self.order = np.delete(np.arange(len(input_shape)), self.freeze_modes)[::-1]

        self.l_fm = len(self.freeze_modes)
        self.bias_rank = bias_rank
        self.method = method
            
        n, m = len(input_shape), len(output_shape)
        l_nfm = n - self.l_fm
        # assert n == m and l_nfm == 3 and np.all(np.array(freeze_modes) < l_nfm), \
        assert n == m and l_nfm == 3, \
            f'Some shape is incorrect: input {n} != output {m}, frozen modes (go first) {l_fm}'
        
        if method == "einsum":
            self.factors = nn.ParameterList([
                nn.Parameter(torch.randn((size_in, size_out))) 
                for s, (size_in, size_out) in enumerate(zip(input_shape, output_shape))
                if s not in freeze_modes
                ])
            self.rule = f'...abc,ad,be,cf->...def'
            # print(f'TCL: {self.rule}', input_shape, output_shape)
        else:
            self.factors = nn.ModuleList([
            nn.Linear(size_in, size_out, bias=False) 
            for s, (size_in, size_out) in enumerate(zip(input_shape, output_shape))
            if s not in self.freeze_modes
            ])

        if bias_rank > 0:
            self.bias_list = nn.ParameterList([
                nn.Parameter(torch.randn(bias_rank, size_out)) 
                for s, size_out in enumerate(output_shape)
                if s not in self.freeze_modes
            ])
            # self.bias = 0
            # for i in range(bias_rank):
            #     self.bias += self.bias_list[0][i][:, None, None] \
            #                * self.bias_list[1][i][None, :, None] \
            #                * self.bias_list[2][i][None, None, :]
            # for f in self.freeze_modes:
            #     self.bias.unsqueeze(f)
        else:
            self.bias_list = []


        self.norm_in  = nn.LayerNorm(input_shape[self.l_fm:])
        self.norm_out = nn.LayerNorm(output_shape[self.l_fm:])

    def forward(self, x) -> torch.Tensor:

        # print(f'TCL forward: {self.rule}, X shape: {x.shape}')
        # print(f'x: {x.device}, w: {self.factors[0].weight.device}, b {self.bias.device}')

        y = self.norm_in(x)

        if self.method == "einsum":
            y = torch.einsum(self.rule, y, *self.factors)
        else:
            y = self.factors[2](x)
            y = self.factors[1](y.transpose(-2, -1))
            y = self.factors[0](y.transpose(-3, -1))
            y = y.permute(*np.arange(self.l_fm), self.l_fm+2, self.l_fm+0, self.l_fm+1)

        for i in range(self.bias_rank):
            y += self.bias_list[0][i][:, None, None] \
               * self.bias_list[1][i][None, :, None] \
               * self.bias_list[2][i][None, None, :]

        y = self.norm_out(y)

        return y

class TCL(nn.Module):
    
    def __init__(self, input_shape, output_shape, bias_rank=True, freeze_modes=None) -> None:
        super().__init__()

        if freeze_modes is None:
            freeze_modes = []

        l_fm = len(freeze_modes)
            
        n, m = len(input_shape), len(output_shape)
        assert n == m and n < 12, \
            f'Some shape is incorrect: input {n} != output {m}'
        
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn((size_in, size_out))) 
            for s, (size_in, size_out) in enumerate(zip(input_shape, output_shape))
            if s not in freeze_modes
            ])

        self.bias = nn.Parameter(torch.randn(output_shape)) if bias_rank else None
        
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

        self.norm_in = nn.LayerNorm(input_shape[l_fm:])
        self.norm_out = nn.LayerNorm(output_shape[l_fm:])

    def forward(self, x) -> torch.Tensor:

        # print(f'TCL forward: {self.rule}, X shape: {x.shape}')
        # print(f'x: {x.device}, w: {self.factors[0].device}')
        x = self.norm_in(x)
        y = torch.einsum(self.rule, x, *self.factors)
        # y = self.norm_out(y)

        # print(f'Output norm {torch.norm(y) / y.shape[0]:.3e}')

        if self.bias is not None:
            y += self.bias

        y = self.norm_out(y)

        return y


class TRL(nn.Module):

    def __init__(self, input_shape, output_shape, core_shape, bias_rank=True, freeze_modes=None) -> None:
        super().__init__()
        
        if freeze_modes is None:
            freeze_modes = []

        l_fm = len(freeze_modes)
        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        assert n + m == l + 2 * l_fm and l < 12, f'' \
            f'Some shape is incorrect: input {n} + output{m} != core {l} + 2 freeze_modes {l_fm}'
        
        input_shape_filtered  = [size for s, size in enumerate(input_shape)  if s not in freeze_modes]
        output_shape_filtered = [size for s, size in enumerate(output_shape) if s not in freeze_modes]

        self.factors_inner = nn.ParameterList([
            nn.Parameter(torch.randn((size_in, size_out))) 
            for s, (size_in, size_out) in enumerate(zip(input_shape_filtered, core_shape[:n-l_fm]))
            ])

        self.factors_outer = nn.ParameterList([
            nn.Parameter(torch.randn((size_in, size_out))) 
            for s, (size_in, size_out) in enumerate(zip(core_shape[n-l_fm:], output_shape_filtered))
            
            ])

        self.core = nn.Parameter(torch.randn(core_shape))
        self.bias = nn.Parameter(torch.randn(output_shape)) if bias_rank else None
        
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

    def forward(self, x) -> torch.Tensor:

        # print(f'TRL forward: {self.rule}, X shape: {x.shape}, {len(self.factors_inner)}, {len(self.factors_outer)}')
        y = torch.einsum(self.rule, x, *self.factors_inner, self.core, *self.factors_outer)

        if self.bias is not None:
            y += self.bias

        return y