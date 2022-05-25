import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl
from typing import List, Tuple

# --------------------------------------------------------------------------------------#
#                                          Loss                                         #
# --------------------------------------------------------------------------------------#

def median_heuristic(med_sqdist, beta=0.5):
    beta_list = [beta ** 2, beta ** 1, 1, (1.0 / beta) ** 1, (1.0 / beta) ** 2]
    return [med_sqdist * b for b in beta_list]

def batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var):
    device = X_p_enc.device
    # some constants, TODO ask Alex
    n_basis = 1024
    gumbel_lmd = 1e+6
    cnst = math.sqrt(1. / n_basis)
    n_mixtures = sigma_var.size(0)
    n_samples = n_basis * n_mixtures
    batch_size, seq_len, nz = X_p_enc.size()

    # gumbel trick to get masking matrix to uniformly sample sigma
    def sample_gmm(W, batch_size):
        U = torch.FloatTensor(batch_size * n_samples, n_mixtures).uniform_()
        U = U.to(W.device)
        sigma_samples = F.softmax(U * gumbel_lmd, dim=1).matmul(sigma_var)
        W_gmm = W.mul(1. / sigma_samples.unsqueeze(1))
        W_gmm = W_gmm.view(batch_size, n_samples, nz)
        return W_gmm

    W = torch.FloatTensor(batch_size * n_samples, nz).normal_(0, 1)
    W = W.to(device)
    W.requires_grad = False
    W_gmm = sample_gmm(W, batch_size)  # batch_size x n_samples x nz
    W_gmm = torch.transpose(W_gmm, 1, 2).contiguous()  # batch_size x nz x n_samples

    XW_p = torch.bmm(X_p_enc, W_gmm)  # batch_size x seq_len x n_samples
    XW_f = torch.bmm(X_f_enc, W_gmm)  # batch_size x seq_len x n_samples
    z_XW_p = cnst * torch.cat((torch.cos(XW_p), torch.sin(XW_p)), 2)
    z_XW_f = cnst * torch.cat((torch.cos(XW_f), torch.sin(XW_f)), 2)
    batch_mmd2_rff = torch.sum((z_XW_p.mean(1) - z_XW_f.mean(1)) ** 2, 1)
    return batch_mmd2_rff


def mmdLossD(X_f,
             Y_f,
             X_f_enc,  # real (initial)   subseq (future window)
             Y_f_enc,  # fake (generated) subseq (future window)
             X_p_enc,  # real (initial)   subseq (past window)
             X_f_dec,
             Y_f_dec,
             lambda_ae,
             lambda_real,
             sigma_var):
    # batchwise MMD2 loss between X_f and Y_f
    D_mmd2 = batch_mmd2_loss(X_f_enc, Y_f_enc, sigma_var)

    # batchwise MMD2 loss between X_p and X_f
    mmd2_real = batch_mmd2_loss(X_p_enc, X_f_enc, sigma_var)

    # reconstruction loss
    real_L2_loss = torch.mean((X_f - X_f_dec) ** 2)
    fake_L2_loss = torch.mean((Y_f - Y_f_dec) ** 2)

    lossD = D_mmd2.mean() - lambda_ae * (real_L2_loss + fake_L2_loss) - lambda_real * mmd2_real.mean()

    return lossD.mean(), mmd2_real.mean()

# --------------------------------------------------------------------------------------#
#                                        Models                                        #
# --------------------------------------------------------------------------------------#

# separation for training
def _history_future_separation(data, window):
    history = data[:, :, :window]
    future = data[:, :, window:2 * window]
    return history, future

class KLCPD(pl.LightningModule):
    def __init__(
            self,
            netG: nn.Module,
            netD: nn.Module,
            args: dict,
            train_dataset: Dataset,
            test_dataset: Dataset,
            num_workers: int = 2
    ) -> None:

        super().__init__()
        self.args = args
        self.netG = netG
        self.netD = netD

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        sigma_list = median_heuristic(self.args['sqdist'], beta=.5)
        self.sigma_var = torch.FloatTensor(sigma_list)

        # to get predictions
        self.window = self.args['window']
        self.num_workers = num_workers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        X = batch[0].to(torch.float32)
        X_p, X_f = _history_future_separation(X, self.args['wnd_dim'])

        X_p = X_p.reshape(-1, self.args['wnd_dim'], self.args['data_dim'])
        X_f = X_f.reshape(-1, self.args['wnd_dim'], self.args['data_dim'])

        batch_size = X_p.size(0)

        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)

        Y_pred = batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var.to(self.device))

        return Y_pred

    # Alternating schedule for optimizer steps (e.g. GANs)
    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: torch.optim.Optimizer,
            optimizer_idx: int,
            optimizer_closure,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False
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
            X_p, X_f = _history_future_separation(X, self.args['wnd_dim'])

            X_p = X_p.reshape(-1, self.args['wnd_dim'], self.args['data_dim'])
            X_f = X_f.reshape(-1, self.args['wnd_dim'], self.args['data_dim'])

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

            lossD, mmd2_real = mmdLossD(X_f, Y_f, X_f_enc, Y_f_enc, X_p_enc, X_f_dec, Y_f_dec,
                                        self.args['lambda_ae'], self.args['lambda_real'],
                                        self.sigma_var.to(self.device))
            lossD = (-1) * lossD
            self.log("train_loss_D", lossD, prog_bar=True)
            self.log("train_mmd2_real_D", mmd2_real, prog_bar=True)
            return lossD

        # optimize generator (netG)
        if optimizer_idx == 0:
            X = batch[0].to(torch.float32)
            X_p, X_f = _history_future_separation(X, self.args['wnd_dim'])

            X_p = X_p.reshape(-1, self.args['wnd_dim'], self.args['data_dim'])
            X_f = X_f.reshape(-1, self.args['wnd_dim'], self.args['data_dim'])

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
            G_mmd2 = batch_mmd2_loss(X_f_enc, Y_f_enc, self.sigma_var.to(self.device))

            lossG = G_mmd2.mean()
            self.log("train_loss_G", lossG, prog_bar=True)

            return lossG

    def validation_step(self,
                        batch: torch.Tensor,
                        batch_idx: int
                        ) -> torch.Tensor:

        X = batch[0].to(torch.float32)
        X_p, X_f = _history_future_separation(X, self.args['wnd_dim'])

        X_p = X_p.reshape(-1, self.args['wnd_dim'], self.args['data_dim'])
        X_f = X_f.reshape(-1, self.args['wnd_dim'], self.args['data_dim'])

        X_p_enc, _ = self.netD(X_p)
        X_f_enc, _ = self.netD(X_f)

        val_mmd2_real = batch_mmd2_loss(X_p_enc, X_f_enc, self.sigma_var.to(self.device))

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


# --------------------------------------------------------------------------------------#
#                                     Predictions                                      #
# --------------------------------------------------------------------------------------#

def _history_future_separation_test(data, window, step=1):
    future_slices = []
    history_slices = []

    if len(data.shape) > 4:
        data = data.transpose(1, 2)

    seq_len = data.shape[1]
    for i in range(0, (seq_len - 2 * window) // step + 1):
        start_ind = i * step
        end_ind = 2 * window + step * i
        slice_2w = data[:, start_ind: end_ind]
        history_slices.append(slice_2w[:, :window].unsqueeze(0))
        future_slices.append(slice_2w[:, window:].unsqueeze(0))

    future_slices = torch.cat(future_slices).transpose(0, 1)
    history_slices = torch.cat(history_slices).transpose(0, 1)

    history_slices = history_slices.transpose(2, 3)
    future_slices = future_slices.transpose(2, 3)

    return history_slices, future_slices

torch.set_printoptions(linewidth=200)
def get_klcpd_output_2(kl_cpd_model, batch, window):
    batch = batch.to(kl_cpd_model.device)
    if len(batch.shape) <= 4:
        seq_len = batch.shape[1]
    else:
        seq_len = batch.shape[2]

    batch_history_slices, batch_future_slices = _history_future_separation_test(batch, window)
    sigma_var = kl_cpd_model.sigma_var.to(kl_cpd_model.device)

    pred_out = []
    for i in range(len(batch_history_slices)):
        zeros = torch.zeros(1, seq_len)

        curr_history = kl_cpd_model.extractor(batch_history_slices[i]).transpose(1, 2).flatten(2)
        curr_history, _ = kl_cpd_model.netD(curr_history.to(torch.float32))

        curr_future = kl_cpd_model.extractor(batch_future_slices[i]).transpose(1, 2).flatten(2)
        curr_future, _ = kl_cpd_model.netD(curr_future.to(torch.float32))

        curr_history, curr_future = [Xi.reshape(*Xi.shape[:2], -1) for Xi in [curr_history, curr_future]]
        mmd_scores = batch_mmd2_loss(curr_history, curr_future, sigma_var)
        zeros[:, 2 * window - 1:] = mmd_scores
        pred_out.append(zeros)
    pred_out = torch.cat(pred_out).to(kl_cpd_model.device)
    # pred_out2 = pred_out[:, 2 * window - 1:].clone().detach().cpu()
    # breakpoint()
    # print(f'batch norm: {pred_out2.norm():.3e}\nnorm: {pred_out2.norm(dim=1)}\nmax : {pred_out2.max(dim=1).values}\nmin : {pred_out2.min(dim=1).values}')
    #TODO fix    
    #TODO check
    # pred_out = pred_out / pred_out.norm(dim=1, keepdim=True)
    #pred_out = torch.softmax(pred_out, dim=1)
    #pred_out = torch.tanh(pred_out)
    # pred_out = torch.tanh(pred_out * 1e4)
    pred_out = torch.tanh(pred_out * 10 ** 7)
    return pred_out