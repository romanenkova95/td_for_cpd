import torch.nn as nn
import pytorch_lightning as pl
import torch
import numpy as np
import random
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
import utils.kl_cpd as klcpd

def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

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

        X = batch[0].to(torch.float32)
        X_p, X_f = klcpd._history_future_separation(X, self.args['wnd_dim'])
        
        X_p = self.extractor(X_p.float())
        X_f = self.extractor(X_f.float())
        
        X_p = X_p.transpose(1, 2).flatten(2) # batch_size, timesteps, C*H*W
        X_f = X_f.transpose(1, 2).flatten(2) # batch_size, timesteps, C*H*W
        
        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)
        
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

            X_p = X_p.transpose(1, 2).flatten(2) # batch_size, timesteps, C*H*W
            X_f = X_f.transpose(1, 2).flatten(2) # batch_size, timesteps, C*H*W
                        
            batch_size = X_p.size(0)

            # real data
            X_p_enc, X_p_dec = self.netD(X_p)
            X_f_enc, X_f_dec = self.netD(X_f)

            # fake data
            noise = torch.FloatTensor(1, batch_size, self.args['RNN_hid_dim']).normal_(0, 1)
            noise.requires_grad = False
            noise = noise.to(self.device)
            
            Y_f = self.netG(X_p, X_f, noise)            
            Y_f = self.netG(X_p, X_f, noise)
            Y_f_enc, Y_f_dec = self.netD(Y_f)
                        
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

            X_p = X_p.transpose(1, 2).flatten(2) # batch_size, timesteps, C*H*W
            X_f = X_f.transpose(1, 2).flatten(2) # batch_size, timesteps, C*H*W
            batch_size = X_p.size(0)
            
            # real data
            X_f_enc, X_f_dec = self.netD(X_f)

            # fake data
            noise = torch.FloatTensor(1, batch_size, self.args['RNN_hid_dim']).normal_(0, 1)
            noise.requires_grad = False
            noise = noise.to(self.device)
            
            Y_f = self.netG(X_p, X_f, noise)
            Y_f_enc, Y_f_dec = self.netD(Y_f)
            
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

        X_p = X_p.transpose(1, 2).flatten(2) # batch_size, timesteps, C*H*W
        X_f = X_f.transpose(1, 2).flatten(2) # batch_size, timesteps, C*H*W

        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)
        
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