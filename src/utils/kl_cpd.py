"""Additional functions for KL-CPD model training and testing."""
from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------------------#
#                                          Loss                                         #
# --------------------------------------------------------------------------------------#
def median_heuristic(med_sqdist: float, beta: float = 0.5) -> List[float]:
    """Initialize kernel's sigma with median heuristic.

    See https://arxiv.org/pdf/1707.07269.pdf

    :param med_sqdist: scale parameter
    :param beta: target sigma
    :return: list of possible sigmas
    """
    beta_list = [beta ** 2, beta ** 1, 1, (1.0 / beta) ** 1, (1.0 / beta) ** 2]
    return [med_sqdist * b for b in beta_list]


def batch_mmd2_loss(
    enc_past: torch.Tensor, enc_future: torch.Tensor, sigma_var: torch.Tensor
) -> object:
    """Calculate MMD loss for batch.

    :param enc_past: encoded past sequence
    :param enc_future: encoded future
    :param sigma_var: tensor with considered kernel parameters
    :return: batch MMD loss
    """
    device = enc_past.device

    # TODO: understand what the constants are
    n_basis = 1024
    gumbel_lmd = 1e6
    norm_cnst = math.sqrt(1.0 / n_basis)
    n_mixtures = sigma_var.size(0)
    n_samples = n_basis * n_mixtures
    batch_size, _, n_latent = enc_past.size()

    weights = (
        torch.FloatTensor(batch_size * n_samples, n_latent).normal_(0, 1).to(device)
    )
    weights.requires_grad = False

    # gumbel trick to get masking matrix to uniformly sample sigma
    uniform_mask = (
        torch.FloatTensor(batch_size * n_samples, n_mixtures).uniform_().to(device)
    )
    sigma_samples = F.softmax(uniform_mask * gumbel_lmd, dim=1).matmul(sigma_var)
    weights_gmm = weights.mul(1.0 / sigma_samples.unsqueeze(1))
    weights_gmm = weights_gmm.reshape(
        batch_size, n_samples, n_latent
    )  # batch_size x n_samples x nz
    weights_gmm = torch.transpose(
        weights_gmm, 1, 2
    ).contiguous()  # batch_size x nz x n_samples

    _kernel_enc_past = torch.bmm(
        enc_past, weights_gmm
    )  # batch_size x seq_len x n_samples
    _kernel_enc_future = torch.bmm(
        enc_future, weights_gmm
    )  # batch_size x seq_len x n_samples

    # approximate kernel with cos and sin
    kernel_enc_past = norm_cnst * torch.cat(
        (torch.cos(_kernel_enc_past), torch.sin(_kernel_enc_past)), 2
    )
    kernel_enc_future = norm_cnst * torch.cat(
        (torch.cos(_kernel_enc_future), torch.sin(_kernel_enc_future)), 2
    )
    batch_mmd2_rff = torch.sum(
        (kernel_enc_past.mean(1) - kernel_enc_future.mean(1)) ** 2, 1
    )
    return batch_mmd2_rff


def mmd_loss_disc(
    input_future: torch.Tensor,
    fake_future: torch.Tensor,
    enc_future: torch.Tensor,
    enc_fake_future: torch.Tensor,
    enc_past: torch.Tensor,
    dec_future: torch.Tensor,
    dec_fake_future: torch.Tensor,
    lambda_ae: float,
    lambda_real: float,
    sigma_var: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate loss for discriminator in KL-CPD model.

    :param input_future: real input subsequence corresponding to the future
    :param fake_future: fake subsequence obtained from generator
    :param enc_future: net_discriminator(input_future)
    :param enc_fake_future: net_discriminator(fake_future)
    :param enc_past: net_discriminator(input_past)
    :param dec_future: last hidden from net_discriminator(input_future)
    :param dec_fake_future: last hidden from net_discriminator(fake_future)
    :param lambda_ae: coefficient before reconstruction loss
    :param lambda_real: coefficient before MMD between past and future
    :param sigma_var: list of sigmas for MMD calculation
    :return: discriminator loss, MMD between real subsequences
    """
    # batch-wise MMD2 loss between real and fake
    mmd2_fake = batch_mmd2_loss(enc_future, enc_fake_future, sigma_var)

    # batch-wise MMD2 loss between past and future
    mmd2_real = batch_mmd2_loss(enc_past, enc_future, sigma_var)

    # reconstruction loss
    real_l2_loss = torch.mean((input_future - dec_future) ** 2)
    fake_l2_loss = torch.mean((fake_future - dec_fake_future) ** 2)

    loss_disc = (
        mmd2_fake.mean()
        - lambda_ae * (real_l2_loss + fake_l2_loss)
        - lambda_real * mmd2_real.mean()
    )

    return loss_disc.mean(), mmd2_real.mean()


# --------------------------------------------------------------------------------------#
#                             Data preprocessing                                        #
# --------------------------------------------------------------------------------------#

# separation for training
def history_future_separation(
    data: torch.Tensor, window: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split sequences in batch on two equal slices.

    :param data: input sequences
    :param window: slice size
    :return: set of "past" subsequences and corresponded "future" subsequences
    """
    history = data[:, :, :window]
    future = data[:, :, window : 2 * window]
    return history, future


# separation for test
def history_future_separation_test(
    data: torch.Tensor,
    window_1: int,
    window_2: Optional[int] = None,
    step: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for testing. Separate it in set of "past"-"future" slices.

    :param data: input sequence
    :param window_1: "past" subsequence size
    :param window_2: "future" subsequence size (default None), if None set equal to window_1
    :param step: step size
    :return: set of "past" subsequences and corresponded "future" subsequences
    """
    future_slices = []
    history_slices = []

    if window_2 is None:
        window_2 = window_1

    if len(data.shape) > 4:
        data = data.transpose(1, 2)

    seq_len = data.shape[1]
    for i in range(0, (seq_len - window_1 - window_2) // step + 1):
        start_ind = i * step
        end_ind = window_1 + window_2 + step * i
        slice_2w = data[:, start_ind:end_ind]
        history_slices.append(slice_2w[:, :window_1].unsqueeze(0))
        future_slices.append(slice_2w[:, window_1:].unsqueeze(0))

    future_slices = torch.cat(future_slices).transpose(0, 1)
    history_slices = torch.cat(history_slices).transpose(0, 1)

    history_slices = history_slices.transpose(2, 3)
    future_slices = future_slices.transpose(2, 3)

    return history_slices, future_slices


# --------------------------------------------------------------------------------------#
#                                     Predictions                                      #
# --------------------------------------------------------------------------------------#


def get_klcpd_output_scaled(
    kl_cpd_model: nn.Module,
    batch: torch.Tensor,
    window_1: int,
    window_2: Optional[int] = None,
    scales: List[float] = [1e10],
) -> List[torch.Tensor]:
    """Get KL-CPD prediction. Scaled it to [0, 1] with scale factors. Try different scales.

    :param kl_cpd_model: pre-trained KL-CPD model
    :param batch: input data
    :param window_1: "past" subsequence size
    :param window_2: "future" subsequence size (default None), if None set equal to window_1
    :param scales: list of scale factors
    :return: list of scaled predictions of MMD scores
    """
    device = kl_cpd_model.device
    batch = batch.to(device)
    sigma_var = kl_cpd_model.sigma_var.to(device)

    # TODO: how to fix
    if len(batch.shape) <= 4:
        seq_len = batch.shape[1]
    else:
        seq_len = batch.shape[2]

    batch_history_slices, batch_future_slices = history_future_separation_test(
        batch, window_1, window_2
    )

    pred_out = []
    for i in range(len(batch_history_slices)):
        zeros = torch.zeros(1, seq_len)


        #curr_history, _ = kl_cpd_model.__get_disc_embeddings(batch_history_slices[i])
        #curr_future, _ = kl_cpd_model.__get_disc_embeddings(batch_future_slices[i])

        #curr_history, curr_future = [
        #    curr_i.reshape(*curr_i.shape[:2], -1)
        #    for curr_i in [curr_history, curr_future]
        #]
        #mmd_scores = batch_mmd2_loss(curr_history, curr_future, sigma_var)
        mmd_scores = kl_cpd_model.__get_disc_embeddings(batch_history_slices[i], batch_future_slices[i])
        zeros[:, window_1 + window_2 - 1 :] = mmd_scores
        pred_out.append(zeros)

    pred_out = torch.cat(pred_out).to(device)
    pred_out_list = [torch.tanh(pred_out * scale) for scale in scales]
    return pred_out_list
