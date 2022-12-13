"""CPD models (KL-CPD and BCE models)."""
from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import utils.kl_cpd as klcpd

class KLCPDVideo(pl.LightningModule):
    """Class for implementation KL-CPD model."""

    def __init__(
        self,
        net_generator: nn.Module,
        net_discriminator: nn.Module,
        args: dict,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_workers: int = 2,
        extractor: nn.Module = None,
    ) -> None:
        """Initialize KL-CPD model.

        :param net_generator: generator model
        :param net_discriminator: discriminator model
        :param args: dictionary with models' parameters
        :param train_dataset: train dataset
        :param test_dataset: test dataset
        :param num_workers: (default 2) number of CPUs
        :param extractor: model for feature extraction, if None use x3d_M 3D_CNN
        """
        super().__init__()
        self.args = args
        self.net_generator = net_generator
        self.net_discriminator = net_discriminator

        if extractor is None:
            # Feature extractor for video datasets
            self.extractor = torch.hub.load(
                "facebookresearch/pytorchvideo:main", "x3d_m", pretrained=True
            )
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))
        else:
            self.extractor = extractor

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = args["batch_size"]

        sigma_list = klcpd.median_heuristic(self.args["sqdist"], beta=0.5)
        self.sigma_var = torch.FloatTensor(sigma_list)

        # to get predictions
        self.window_1 = self.args["window_1"]
        self.window_2 = self.args["window_2"]

        self.num_workers = num_workers

        self.masked = args["block_type"] == "masked"
        if self.masked:
            self.alpha_disc, self.alpha_gen = args["alphaD"], args["alphaG"]

    def __apply_extractor(self, input):
        input = self.extractor(input.float())  # batch_size, C, seq_len, H, W
        input = input.transpose(1, 2)#.flatten(2)
        return input

    def __initialize_noise(self, batch_size):
        if np.isscalar(self.args["rnn_hid_dim"]):
            noise = torch.FloatTensor(1, batch_size, self.args["rnn_hid_dim"]).normal_(
                0, 1
            )
        else:
            noise = torch.FloatTensor(batch_size, *self.args["rnn_hid_dim"]).normal_(
                0, 1
            )

        noise.requires_grad = False
        noise = noise.to(self.device)
        return noise

    def __get_disc_embeddings(self, input_past: torch.Tensor,
                              input_future: torch.Tensor) -> torch.Tensor:
        input_past = self.__apply_extractor(input_past)
        input_future = self.__apply_extractor(input_future)

        enc_past, _ = self.net_discriminator(input_past)
        enc_future, _ = self.net_discriminator(input_future)

        enc_past, enc_future = [
            enc_i.reshape(*enc_i.shape[:2], -1) for enc_i in [enc_past, enc_future]
        ]
        predicted_mmd_score = klcpd.batch_mmd2_loss(
            enc_past, enc_future, self.sigma_var.to(self.device)
        )
        return predicted_mmd_score

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for KL-CPD model.

        :param inputs: input data
        :return: embedded data
        """
        input_past, input_future = klcpd.history_future_separation(
            inputs[0].to(torch.float32), self.args["wnd_dim"]
        )
        predicted_mmd_score = self.__get_disc_embeddings(input_past, input_future)
        return predicted_mmd_score

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        optimizer_idx: int,
        optimizer_closure,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ):
        """Define optimization step for KL-CPD model.

        :param epoch: number of current epoch
        :param batch_idx: number of used batch index
        :param optimizer: used optimizers (for generator and discriminator)
        :param optimizer_idx: if 0 - update generator, if 1 - update discriminator
        :param optimizer_closure: closure
        :param on_tpu: if True calculate on TPU
        :param using_native_amp: some parameters
        :param using_lbfgs: some parameters
        """
        # update generator every CRITIC_ITERS steps
        if optimizer_idx == 0:
            if (batch_idx + 1) % self.args["CRITIC_ITERS"] == 0:
                # the closure (which includes the `training_step`) will be
                # executed by `optimizer.step`
                optimizer.step(closure=optimizer_closure)
            else:
                # call the closure by itself to run `training_step` + `backward`
                # without an optimizer step
                optimizer_closure()

        # update discriminator every step
        if optimizer_idx == 1:
            for param in self.net_discriminator.rnn_enc_layer.parameters():
                param.data.clamp_(-self.args["weight_clip"], self.args["weight_clip"])
            optimizer.step(closure=optimizer_closure)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        """Training step for KL-CPD model.

        :param batch: input data
        :param batch_idx: index of batch
        :param optimizer_idx: index of optimizer (0 for generator, 1 for discriminator)
        :return: train loss
        """
        batch_size = batch[0].size(0)
        input_past, input_future = klcpd.history_future_separation(
            batch[0].to(torch.float32), self.args["wnd_dim"]
        )
        input_past = self.__apply_extractor(input_past)
        input_future = self.__apply_extractor(input_future)

        enc_past, _ = self.net_discriminator(input_past)
        enc_future, hidden_future = self.net_discriminator(input_future)
        noise = self.__initialize_noise(batch_size)

        fake_future = self.net_generator(input_past, input_future, noise)
        enc_fake_future, hidden_fake_future = self.net_discriminator(fake_future)
        # optimize discriminator
        if optimizer_idx == 1:
            all_data = [
                input_future,
                fake_future,
                enc_future,
                enc_fake_future,
                enc_past,
                hidden_future,
                hidden_fake_future,
            ]
            all_data = [data_i.reshape(*data_i.shape[:2], -1) for data_i in all_data]

            loss_disc, mmd2_real = klcpd.mmd_loss_disc(
                *all_data,
                self.args["lambda_ae"],
                self.args["lambda_real"],
                self.sigma_var.to(self.device)
            )
            loss_disc = (-1) * loss_disc
            self.log("tlD", loss_disc, prog_bar=True)
            self.log("train_mmd2_real_D", mmd2_real, prog_bar=True)

            if self.masked:
                mask_loss_disc = self.net_discriminator.mask_loss()
                self.log("mlD", mask_loss_disc, prog_bar=True)
                loss_disc += self.alpha_disc * mask_loss_disc

            return loss_disc

        # optimize generator
        if optimizer_idx == 0:
            all_future_enc = [enc_future, enc_fake_future]
            all_future_enc = [
                enc_i.reshape(*enc_i.shape[:2], -1) for enc_i in all_future_enc
            ]

            # batch-wise MMD2 loss between input_future and fake_future
            gen_mmd2 = klcpd.batch_mmd2_loss(
                *all_future_enc, self.sigma_var.to(self.device)
            )
            loss_gen = gen_mmd2.mean()
            self.log("tlG", loss_gen, prog_bar=True)

            return loss_gen

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Training step for KL-CPD model.

        :param batch: input data
        :param batch_idx: index of batch
        :return: MMD score
        """

        val_mmd2_real = self.forward(batch)
        self.log("val_mmd2_real_D", val_mmd2_real, prog_bar=True)

        return val_mmd2_real

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Configure optimizers.

        :return: optimizers
        """
        optimizer_gen = torch.optim.Adam(
            self.net_generator.parameters(),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )

        optimizer_disc = torch.optim.Adam(
            self.net_discriminator.parameters(),
            lr=self.args["lr"],
            weight_decay=self.args["weight_decay"],
        )

        return optimizer_gen, optimizer_disc

    def train_dataloader(self) -> DataLoader:
        """Set train dataloader.

        :return: dataloader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Set val dataloader.

        :return: dataloader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


#################################################################
class CPDModel(pl.LightningModule):
    """Pytorch Lightning wrapper for change point detection models."""

    def __init__(
        self,
        model: nn.Module,
        args: dict,
        train_dataset,
        test_dataset,
        num_workers: int = 2,
        extractor=None,
    ) -> None:
        """Initialize CPD model.

        :param model: custom base model
        :param num_workers: num of kernels used for evaluation
        """
        super().__init__()

        self.model = model

        self.learning_rate = args["lr"]
        self.batch_size = args["batch_size"]
        self.num_workers = num_workers

        self.loss = nn.BCELoss()

        self.train_dataset, self.test_dataset = train_dataset, test_dataset

        if extractor is None:
            # Feature extractor for video datasets
            self.extractor = torch.hub.load(
                "facebookresearch/pytorchvideo:main", "x3d_m", pretrained=True
            )
            self.extractor = nn.Sequential(*list(self.extractor.blocks[:5]))
        else:
            self.extractor = extractor

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward step for CPD model.

        :param inputs: batch of data
        :return: predictions
        """
        return self.model(self.extractor(inputs).transpose(1, 2))

    def training_step(self, batch: torch.Tensor, *args) -> torch.Tensor:
        """Train CPD model.

        :param batch: data for training
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())
        train_loss = self.loss(pred.squeeze(), labels.squeeze().float())

        train_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_acc", train_accuracy, prog_bar=True)

        return train_loss

    def validation_step(self, batch: torch.Tensor, *args) -> torch.Tensor:
        """Test CPD model.

        :param batch: data for validation
        :return: loss function value
        """
        inputs, labels = batch
        pred = self.forward(inputs.float())

        val_loss = self.loss(pred.squeeze(), labels.squeeze().float())
        val_accuracy = (
            ((pred.squeeze() > 0.5).long() == labels.squeeze()).float().mean()
        )

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", val_accuracy, prog_bar=True)

        return val_loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initialize optimizer.

        :return: optimizer for training CPD model
        """
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return opt

    def train_dataloader(self) -> DataLoader:
        """Initialize dataloader for training.

        :return: dataloader for training
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """Initialize dataloader for validation.

        :return: dataloader for validation
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
