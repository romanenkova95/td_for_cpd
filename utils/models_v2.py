import torch.nn as nn
import pytorch_lightning as pl
import torch
import numpy as np
import random
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
import utils.kl_cpd as klcpd
import string

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

class TCL(nn.Module):
    
    def __init__(self, input_shape, output_shape, has_bias=True, freeze_modes=None) -> None:

        super().__init__()
            
        n, m = len(input_shape), len(output_shape)
        assert n == m and n < 12, \
            f'Some shape is incorrect: input {n} != output {m}'
        
        self.factors = nn.ParameterList([
            nn.Parameter(torch.randn((size_in, size_out))) 
            for s, (size_in, size_out) in enumerate(zip(input_shape, output_shape))
            if s not in freeze_modes
            ])

        self.bias = nn.Parameter(torch.randn(output_shape)) if has_bias else None
        
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

    def forward(self, x) -> torch.Tensor:

        # print(f'TCL forward: {self.rule}, X shape: {x.shape}')
        # print(f'x: {x.device}, w: {self.factors[0].device}')
        y = torch.einsum(self.rule, x, *self.factors)

        if self.bias is not None:
            y += self.bias

        return y

class TRL(nn.Module):

    def __init__(self, input_shape, output_shape, core_shape, has_bias=True) -> None:

        super().__init__()
        n, m, l = len(input_shape), len(output_shape), len(core_shape)
        assert n + m == l and l < 12, f'' \
            f'Some shape is incorrect: input {n} + output{m} != core {l}'
        
        self.factors_inner = [
            nn.Parameter(torch.randn((size_in, size_out))) 
            for size_in, size_out in zip(input_shape, core_shape[:n])
            ]
        self.factors_outer = [
            nn.Parameter(torch.randn((size_in, size_out))) 
            for size_in, size_out in zip(core_shape[n:], output_shape)
            ]
        self.core = nn.Parameter(torch.randn(core_shape))
        self.bias = nn.Parameter(torch.randn(output_shape)) if has_bias else None
        
        chars_inner = string.ascii_lowercase[:n]
        chars_core  = string.ascii_lowercase[n:2*n]
        chars_outer = string.ascii_lowercase[2*n:2*n+m]
        chars_final = string.ascii_lowercase[2*n+m:2*n+2*m]

        self.rule = f'{chars_inner},' \
                    f'{",".join([i+j for i, j in zip(chars_inner, chars_core)])}' \
                    f',{chars_core}{chars_outer},' \
                    f'{",".join([i+j for i, j in zip(chars_outer, chars_final)])}' \
                    f'->{chars_final}'

        print(f'TRL: {self.rule}')

    def forward(self, x) -> torch.Tensor:

        y = torch.einsum(self.rule, x, *self.factors_inner, self.core, *self.factors_outer)

        if self.bias is not None:
            y += self.bias

        return y
    
class custom_GRU_TCL(nn.Module):

    # TODO add num_layers parameter
    def __init__(self, input_dims, hidden_dims,  has_bias=False, freeze_modes=None):
        super().__init__()
        
        self.hidden_dim = hidden_dims
        self.linear_w = nn.ModuleList([
            TCL(input_dims, hidden_dims,  has_bias, freeze_modes) 
            for _ in range(3)])
        
        self.linear_u = nn.ModuleList([
            TCL(hidden_dims, hidden_dims,  has_bias, freeze_modes) 
            for _ in range(3)])
        
    
    def forward(self, inputs, h_prev=None):

        # inputs shape: L, N, Hin
        L, N, *input_dims = inputs.shape

        outputs = []
        if h_prev is None:
            h_prev = torch.zeros(N, *self.hidden_dim[1:]).to(inputs)
        
        # print(f'h0: {h_prev.shape}')
        
        for x_t in inputs:
            x_z, x_r, x_h = [linear(x_t) for linear in self.linear_w]
            h_z, h_r, h_h = [linear(h_prev) for linear in self.linear_u]
            
            output_z = torch.sigmoid(x_z + h_z)
            output_r = torch.sigmoid(x_r + h_r)
            hidden_hat = torch.tanh(x_h + output_r*h_h)
            h_prev = output_z * h_prev + (1 - output_z) * hidden_hat

            outputs.append(h_prev)#.clone().detach()) # FIXME check squeeze dim
        
        outputs = torch.stack(outputs, dim=0)
        # print(f'h1: {h_prev.shape}')

        return outputs, h_prev
    
class NetD_TCL(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.RNN_hid_dims = args['RNN_hid_dims']
        self.emb_dims = args['emb_dims']
        
        self.fc1 = TCL((1, 1,) + args['data_dims'], 
                       (1, 1,) + self.emb_dims, 
                       has_bias=False, 
                       freeze_modes=[0, 1])

        self.rnn_enc_layer = custom_GRU_TCL((1,) + self.emb_dims, 
                                            (1,) + self.RNN_hid_dims, 
                                            has_bias=False, 
                                            freeze_modes=[0]) 
                                            #, num_layers=args['num_layers'], batch_first=True)
        
        self.rnn_dec_layer = custom_GRU_TCL((1,) + self.RNN_hid_dims, 
                                            (1,) + self.emb_dims, 
                                            has_bias=False, 
                                            freeze_modes=[0]) 
                                            #, num_layers=args['num_layers'], batch_first=True)

        self.fc2 = TCL((1, 1,) + self.emb_dims, 
                       (1, 1,) + args['data_dims'], 
                       has_bias=False, 
                       freeze_modes=[0, 1])
        self.relu = nn.ReLU()
        
    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor]:

        X = X.transpose(0, 1) # sequence first (L, N, input_dims)
        X = self.relu(self.fc1(X))
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        X_dec = self.relu(self.fc2(X_dec))
        X_enc = X_enc.transpose(0, 1) # batch first (L, N, input_dims)
        X_dec = X_dec.transpose(0, 1) # batch first (L, N, input_dims)
        return X_enc, X_dec
    

class NetG_TCL(nn.Module):
    def __init__(self, args) -> None:
            
        super().__init__()
        self.RNN_hid_dims = args['RNN_hid_dims']
        self.emb_dims = args['emb_dims']
        self.relu = nn.ReLU()
        
        self.fc = TCL((1, 1,) + args['data_dims'], 
                      (1, 1,) + self.emb_dims, 
                      has_bias=False, 
                      freeze_modes=[0, 1])

        self.rnn_enc_layer = custom_GRU_TCL((1,) + self.emb_dims, 
                                            (1,) + self.RNN_hid_dims, 
                                            has_bias=False, 
                                            freeze_modes=[0]) 
                                            #, num_layers=args['num_layers'], batch_first=True)
        
        self.rnn_dec_layer = custom_GRU_TCL((1,) + self.emb_dims,
                                            (1,) + self.RNN_hid_dims,  
                                            has_bias=False, 
                                            freeze_modes=[0]) 
                                            #, num_layers=args['num_layers'], batch_first=True)

        self.fc_layer = TCL((1, 1,) + self.RNN_hid_dims, 
                       (1, 1,) + args['data_dims'], 
                       has_bias=False, 
                       freeze_modes=[0, 1])

        # self.rnn_enc_layer = nn.GRU(self.emb_dim, self.RNN_hid_dim, num_layers=args['num_layers'], batch_first=True)
        # self.rnn_dec_layer = nn.GRU(self.emb_dim, self.RNN_hid_dim, num_layers=args['num_layers'], batch_first=True)
        # self.fc_layer = nn.Linear(self.RNN_hid_dim, args['data_dim'])

    
    def forward(self, X_p, X_f, noise) -> torch.Tensor:

        X_p = X_p.transpose(0, 1) # sequence first (L, N, input_dims)
        X_p = self.relu(self.fc(X_p))

        X_f = X_f.transpose(0, 1) # sequence first (L, N, input_dims)
        X_f = self.relu(self.fc(X_f))

        # FIXME continue from here
        _, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        output = output.transpose(0, 1) # batch first (L, N, input_dims)
        return output

    def shft_right_one(self, X) -> torch.Tensor:
        X_shft = X.clone()
        X_shft[0].data.fill_(0)
        X_shft[1:] = X[:-1]
        return X_shft


class NetG(nn.Module):
    def __init__(self, args) -> None:
            
        super(NetG, self).__init__()
        self.RNN_hid_dim = args['RNN_hid_dim']
        self.emb_dim = args['emb_dim']
        self.relu = nn.ReLU()
        
        self.fc = nn.Linear(args['data_dim'], self.emb_dim)
        self.rnn_enc_layer = nn.GRU(self.emb_dim, self.RNN_hid_dim, num_layers=args['num_layers'], batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.emb_dim, self.RNN_hid_dim, num_layers=args['num_layers'], batch_first=True)
        self.fc_layer = nn.Linear(self.RNN_hid_dim, args['data_dim'])

    
    def forward(self, X_p, X_f, noise) -> torch.Tensor:
        X_p = self.relu(self.fc(X_p))
        X_f = self.relu(self.fc(X_f))

        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)
        return output

    def shft_right_one(self, X) -> torch.Tensor:
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft
    
    
class NetD(nn.Module):
    def __init__(self, args) -> None:
        super(NetD, self).__init__()
        self.RNN_hid_dim = args['RNN_hid_dim']
        self.emb_dim = args['emb_dim']
        
        self.fc1 = nn.Linear(args['data_dim'], self.emb_dim)

        self.rnn_enc_layer = nn.GRU(self.emb_dim, self.RNN_hid_dim, num_layers=args['num_layers'], batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.emb_dim, num_layers=args['num_layers'], batch_first=True)

        self.fc2 = nn.Linear(self.emb_dim, args['data_dim'])
        self.relu = nn.ReLU()
        
    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.relu(self.fc1(X))
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        X_dec = self.relu(self.fc2(X_dec))
        return X_enc, X_dec

#################################################################    
class KLCPDVideo(pl.LightningModule):
    def __init__(
        self,
        netG: nn.Module,
        netD: nn.Module,
        args: dict,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_workers: int=2
    ) -> None:

        super().__init__()        
        self.args = args
        self.netG = netG
        self.netD = netD
        
        # Feature extractor for video datasets
        self.extractor = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
        self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        sigma_list = klcpd.median_heuristic(self.args['sqdist'], beta=.5)
        self.sigma_var = torch.FloatTensor(sigma_list)
        
        # to get predictions
        self.window_1 = self.args['window_1']
        self.window_2 = self.args['window_2']
        
        self.num_workers = num_workers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        X = inputs[0].to(torch.float32)
        X_p, X_f = klcpd._history_future_separation(X, self.args['wnd_dim'])
        
        X_p = self.extractor(X_p.float())
        X_f = self.extractor(X_f.float())
        
        X_p = X_p.transpose(1, 2) #.flatten(2) # batch_size, timesteps, C*H*W
        X_f = X_f.transpose(1, 2) #.flatten(2) # batch_size, timesteps, C*H*W
        # print(f'X_p {X_p.shape}')
        
        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)
        
        X_p_enc, X_f_enc = [Xi.reshape(*Xi.shape[:2], -1) for Xi in [X_p_enc, X_f_enc]]
        Y_pred = klcpd.batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var.to(self.device))
        
        return Y_pred
    
    # Alternating schedule for optimizer steps (e.g. GANs)
    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure,
        on_tpu: bool=False,
        using_native_amp: bool=False,
        using_lbfgs: bool=False
    ):
        # update generator every CRITIC_ITERS steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.args['CRITIC_ITERS'] == 0:
                # the closure (which includes the `training_step`) will be executed by `optimizer.step`
                optimizer.step(closure=optimizer_closure)
            else:
                # call the closure by itself to run `training_step` + `backward` without an optimizer step
                optimizer_closure()
        
        # update discriminator every step
        if optimizer_idx == 1:
            for p in self.netD.rnn_enc_layer.parameters():
                p.data.clamp_(-self.args['weight_clip'], self.args['weight_clip'])
            optimizer.step(closure=optimizer_closure)
        
    def training_step(self,
                      batch: torch.Tensor,
                      batch_idx: int,
                      optimizer_idx: int
                     ) -> torch.Tensor:
        
        # optimize discriminator (netD)
        if optimizer_idx == 1:
            X = batch[0].to(torch.float32)
            X_p, X_f = klcpd._history_future_separation(X, self.args['wnd_dim'])
            
            X_p = self.extractor(X_p.float())
            X_f = self.extractor(X_f.float())

            X_p = X_p.transpose(1, 2)#.flatten(2) # batch_size, timesteps, C*H*W
            X_f = X_f.transpose(1, 2)#.flatten(2) # batch_size, timesteps, C*H*W
                        
            batch_size = X_p.size(0)

            # real data
            X_p_enc, X_p_dec = self.netD(X_p)
            X_f_enc, X_f_dec = self.netD(X_f)

            # fake data
            # noise = torch.FloatTensor(1, batch_size, self.args['RNN_hid_dims']).normal_(0, 1)
            noise = torch.FloatTensor(batch_size, *self.args['RNN_hid_dims']).normal_(0, 1)
            noise.requires_grad = False
            noise = noise.to(self.device)
            
            # Y_f = self.netG(X_p, X_f, noise)            
            Y_f = self.netG(X_p, X_f, noise)
            Y_f_enc, Y_f_dec = self.netD(Y_f)
                        
            X_f, Y_f, X_f_enc, Y_f_enc, X_p_enc, X_f_dec, Y_f_dec = [
                Xi.reshape(*Xi.shape[:2], -1) 
                for Xi in [X_f, Y_f, X_f_enc, Y_f_enc, X_p_enc, X_f_dec, Y_f_dec]]

            lossD, mmd2_real = klcpd.mmdLossD(X_f, Y_f, X_f_enc, Y_f_enc, X_p_enc, X_f_dec, Y_f_dec,
                                              self.args['lambda_ae'], self.args['lambda_real'],
                                              self.sigma_var.to(self.device))
            lossD = (-1) * lossD
            self.log("train_loss_D", lossD, prog_bar=True)
            self.log("train_mmd2_real_D", mmd2_real, prog_bar=True)
            
            #print('train loss D:', lossD)
            
            return lossD
            
        # optimize generator (netG)
        if optimizer_idx == 0:
            X = batch[0].to(torch.float32)
            X_p, X_f = klcpd._history_future_separation(X, self.args['wnd_dim'])
            
            X_p = self.extractor(X_p.float())
            X_f = self.extractor(X_f.float())

            X_p = X_p.transpose(1, 2)#.flatten(2) # batch_size, timesteps, C*H*W
            X_f = X_f.transpose(1, 2)#.flatten(2) # batch_size, timesteps, C*H*W
            batch_size = X_p.size(0)
            
            # real data
            X_f_enc, X_f_dec = self.netD(X_f)

            # fake data
            # noise = torch.FloatTensor(1, batch_size, self.args['RNN_hid_dims']).normal_(0, 1)
            noise = torch.FloatTensor(batch_size, *self.args['RNN_hid_dims']).normal_(0, 1)
            noise.requires_grad = False
            noise = noise.to(self.device)
            
            Y_f = self.netG(X_p, X_f, noise)
            Y_f_enc, Y_f_dec = self.netD(Y_f)
            
            X_f_enc, Y_f_enc = [Xi.reshape(*Xi.shape[:2], -1) for Xi in [X_f_enc, Y_f_enc]]
            # batchwise MMD2 loss between X_f and Y_f
            G_mmd2 = klcpd.batch_mmd2_loss(X_f_enc, Y_f_enc, self.sigma_var.to(self.device))
            
            lossG = G_mmd2.mean()
            self.log("train_loss_G", lossG, prog_bar=True)
            
            #print('train loss G:', lossG)
            
            return lossG
    
    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int
                       ) -> torch.Tensor:
    
        X = batch[0].to(torch.float32)
        X_p, X_f = klcpd._history_future_separation(X, self.args['wnd_dim'])
        
        X_p = self.extractor(X_p.float())
        X_f = self.extractor(X_f.float())

        # print(f'X_p device: {X_p.device}')
        X_p = X_p.transpose(1, 2)#.flatten(2) # batch_size, timesteps, C*H*W
        X_f = X_f.transpose(1, 2)#.flatten(2) # batch_size, timesteps, C*H*W

        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)
        
        X_p_enc, X_f_enc = [Xi.reshape(*Xi.shape[:2], -1) for Xi in [X_p_enc, X_f_enc]]
        val_mmd2_real = klcpd.batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var.to(self.device))
    
        self.log('val_mmd2_real_D', val_mmd2_real, prog_bar=True)

        return val_mmd2_real
    
    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        
        optimizerG = torch.optim.Adam(self.netG.parameters(), 
                                      lr=self.args['lr'], 
                                      weight_decay=self.args['weight_decay'])
        
        optimizerD = torch.optim.Adam(self.netD.parameters(),
                                      lr=self.args['lr'], 
                                      weight_decay=self.args['weight_decay'])
        
        return optimizerG, optimizerD
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args['batch_size'], shuffle=True, 
                          num_workers=self.num_workers)
        
    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args['batch_size'], shuffle=False,
                          num_workers=self.num_workers)
        
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args['batch_size'], shuffle=False,
                          num_workers=self.num_workers)    