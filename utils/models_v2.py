import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import random
from typing import List, Tuple
from torch.utils.data import DataLoader, Dataset
import utils.kl_cpd as klcpd
import utils.datasets as datasets


def fix_seeds(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

#################################################################
class KLCPDVideo(pl.LightningModule):
    def __init__(
        self,
        netG: nn.Module,
        netD: nn.Module,
        args: dict,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_workers: int=2,
        extractor: nn.Module=None
    ) -> None:

        super().__init__()
        self.args = args
        self.netG = netG
        self.netD = netD

        if extractor == None:
            # Feature extractor for video datasets
            self.extractor = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))
        else:
            self.extractor = extractor

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        sigma_list = klcpd.median_heuristic(self.args['sqdist'], beta=.5)
        self.sigma_var = torch.FloatTensor(sigma_list)

        # to get predictions
        self.window_1 = self.args['window_1']
        self.window_2 = self.args['window_2']

        self.num_workers = num_workers

        self.masked = args["block_type"] == "masked"
        if self.masked:
            self.alphaD, self.alphaG = args["alphaD"], args["alphaG"]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        X = inputs[0].to(torch.float32)
        X_p, X_f = klcpd._history_future_separation(X, self.args['wnd_dim'])

        X_p = self.extractor(X_p.float())
        X_f = self.extractor(X_f.float())

        # print(torch.norm(X_p) / X_p.shape[0], torch.norm(X_f) / X_f.shape[0])

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
            # noise = torch.FloatTensor(1, batch_size, self.args['RNN_hid_dim']).normal_(0, 1)
            if np.isscalar(self.args['RNN_hid_dim']):
                noise = torch.FloatTensor(1, batch_size, self.args['RNN_hid_dim']).normal_(0, 1)
            else:
                noise = torch.FloatTensor(batch_size, *self.args['RNN_hid_dim']).normal_(0, 1)
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
            self.log("tlD", lossD, prog_bar=True)
            self.log("train_mmd2_real_D", mmd2_real, prog_bar=True)

            if self.masked:
                mask_loss_D = self.netD.mask_loss()
                self.log("mlD", mask_loss_D, prog_bar=True)
                lossD += self.alphaD * mask_loss_D

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
            # noise = torch.FloatTensor(1, batch_size, self.args['RNN_hid_dim']).normal_(0, 1)
            # noise = torch.FloatTensor(batch_size, *self.args['RNN_hid_dim']).normal_(0, 1)
            if np.isscalar(self.args['RNN_hid_dim']):
                noise = torch.FloatTensor(1, batch_size, self.args['RNN_hid_dim']).normal_(0, 1)
            else:
                noise = torch.FloatTensor(batch_size, *self.args['RNN_hid_dim']).normal_(0, 1)

            noise.requires_grad = False
            noise = noise.to(self.device)

            Y_f = self.netG(X_p, X_f, noise)
            Y_f_enc, Y_f_dec = self.netD(Y_f)

            X_f_enc, Y_f_enc = [Xi.reshape(*Xi.shape[:2], -1) for Xi in [X_f_enc, Y_f_enc]]
            # batchwise MMD2 loss between X_f and Y_f
            G_mmd2 = klcpd.batch_mmd2_loss(X_f_enc, Y_f_enc, self.sigma_var.to(self.device))

            lossG = G_mmd2.mean()
            self.log("tlG", lossG, prog_bar=True)

            # if self.masked:
            #     mask_loss_G = self.netG.mask_loss()
            #     self.log("mlG", mask_loss_G, prog_bar=True)
            #     lossG += self.alphaG * mask_loss_G

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

#################################################################
class CPD_model(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        model: nn.Module,
        args: dict,
        train_dataset,
        test_dataset,
        lr: float = 1e-3,
        batch_size: int = 64,
        num_workers: int = 2,
        extractor = None,
    ) -> None:
        """Initialize CPD model.
        :param experiment_type: type of data used for training
        :param loss_type: type of loss function for training special CPD or common BCE loss
        :param T: parameter restricted the size of a considered segment in delay loss (T in the paper)
        :param model: custom base model (or None if need a default one)
        :param lr: learning rate
        :param batch_size: size of batch
        :param num_workers: num of kernels used for evaluation
        """
        super().__init__()

        self.model = model

        self.lr = args["lr"]
        self.batch_size = args["batch_size"]
        self.num_workers = num_workers

        self.loss = nn.BCELoss()


        self.train_dataset, self.test_dataset = train_dataset, test_dataset

        if extractor == None:
            # Feature extractor for video datasets
            self.extractor = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))
        else:
            self.extractor = extractor


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for CPD model.

        :param inputs: batch of data
        :return: predictions
        """
        return self.model(self.extractor(inputs).transpose(1, 2))

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Train CPD model.

        :param batch: data for training
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())
        train_loss = self.loss(pred.squeeze(), labels.squeeze().float())

        train_accuracy = (((pred.squeeze() >
                            0.5).long() == labels.squeeze()).float().mean())


        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return train_loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Test CPD model.

        :param batch: data for validation
        :param batch_idx: index of batch (special for pytorch-lightning)
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        val_loss = self.loss(pred.squeeze(), labels.squeeze().float())
        val_accuracy = (((pred.squeeze() >
                          0.5).long() == labels.squeeze()).float().mean())

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Initialize dataloader for training.

        :return: dataloader for training
        """
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        """Initialize dataloader for test (same as for validation).

        :return: dataloader for test
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
