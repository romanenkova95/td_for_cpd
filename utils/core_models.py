"""Core models for CPD."""
# TODO add masked

from typing import Dict, Tuple

import random
import torch
import torch.nn as nn
import numpy as np

from .tl_base_cells import init_fc_rnn_tl, init_bias, GruTl, parse_bce_args


def fix_seeds(seed: int) -> None:
    """Fix seeds for experiment reproduction.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


########################KL-CPD######################################
class KlcpdGen(nn.Module):
    """Class for KL-CPD generator."""

    def __init__(self, args) -> None:
        """Initialize generator.

        :param args: dict with the network parameters:
            - data dimensions
            - embedding dimensions
            - hidden dimension for RNN
        """
        super().__init__()
        self.rnn_hid_dim = args["rnn_hid_dim"]
        self.emb_dim = args["emb_dim"]
        self.relu = nn.ReLU()

        self.emb_layer = nn.Linear(args["data_dim"], self.emb_dim)
        self.rnn_enc_layer = nn.GRU(
            self.emb_dim,
            self.rnn_hid_dim,
            num_layers=args["num_layers"],
            batch_first=True,
        )
        self.rnn_dec_layer = nn.GRU(
            self.emb_dim,
            self.rnn_hid_dim,
            num_layers=args["num_layers"],
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.rnn_hid_dim, args["data_dim"])

    def forward(
        self, past_input: torch.Tensor, future_input: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward step for the generator.

        :param past_input: input data from past
        :param future_input: input data from future
        :param noise: input noise
        :return: embedded data
        """
        past_input = past_input.flatten(2)  # batch_size, timesteps, C*H*W
        future_input = future_input.flatten(2)  # batch_size, timesteps, C*H*W

        past_input = self.relu(self.emb_layer(past_input))
        future_input = self.relu(self.emb_layer(future_input))

        _, h_t = self.rnn_enc_layer(past_input)

        future_inp_shift = future_input.clone()
        future_inp_shift[:, 0, :].data.fill_(0)
        future_inp_shift[:, 1:, :] = future_input[:, :-1, :]

        hidden = h_t + noise
        future_target, _ = self.rnn_dec_layer(future_inp_shift, hidden)
        output = self.output_layer(future_target)
        return output


class KlcpdDisc(nn.Module):
    """Class for KL-CPD discriminator."""

    def __init__(self, args) -> None:
        """Initialize discriminator.

        :param args: dict with the network parameters:
            - data dimensions
            - embedding dimensions
            - hidden dimension for RNN
        """
        super().__init__()
        self.rnn_hid_dim = args["rnn_hid_dim"]
        self.emb_dim = args["emb_dim"]

        self.emb_layer = nn.Linear(args["data_dim"], self.emb_dim)

        self.rnn_enc_layer = nn.GRU(
            self.emb_dim,
            self.rnn_hid_dim,
            num_layers=args["num_layers"],
            batch_first=True,
        )
        self.rnn_dec_layer = nn.GRU(
            self.rnn_hid_dim,
            self.emb_dim,
            num_layers=args["num_layers"],
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.emb_dim, args["data_dim"])
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward step for the discriminator.

        :param input: input data
        :return: encoded and decoded inputs
        """
        input = input.flatten(2)
        output = self.relu(self.emb_layer(input))
        output_enc, _ = self.rnn_enc_layer(output)
        output_dec, _ = self.rnn_dec_layer(output_enc)
        output_dec = self.relu(self.output_layer(output_dec))

        return output_enc, output_dec


class KlcpdGenTl(nn.Module):
    """Class for KL-CPD generator with tensor-layer decomposition."""

    def __init__(self, args) -> None:
        """Initialize generator with tensor-layer decomposition.

        :param args: dict with the network parameters:
            - type of tensor-layer decomposition
            - data dimensions
            - embedding dimensions
            - hidden dimension for RNN
            - bias rank for tensor-layer
        """
        super().__init__()
        block_type = args["block_type"]
        self.rnn_hid_dims = args["rnn_hid_dim"]
        self.emb_dims = args["emb_dim"]
        self.relu = nn.ReLU()
        feature_dims = args["feature_dim"] 

        fc_bias, gru_bias = init_bias(args)
        block, (args_in, args_gru, args_out) = init_fc_rnn_tl(block_type, args)

        self.emb_layer = block(
            (1, 1,) + args["data_dim"],
            (1, 1,) + self.emb_dims,
            bias_rank=fc_bias,
            feature_dims=feature_dims,
            **args_in
        )

        self.rnn_enc_layer = GruTl(
            block_type,
            (1,) + self.emb_dims,
            (1,) + self.rnn_hid_dims,
            bias_rank=gru_bias,
            feature_dims=feature_dims,
            **args_gru
        )

        self.rnn_dec_layer = GruTl(
            block_type,
            (1,) + self.emb_dims,
            (1,) + self.rnn_hid_dims,
            bias_rank=gru_bias,
            feature_dims=feature_dims,
            **args_gru
        )

        self.output_layer = block(
            (1, 1,) + self.rnn_hid_dims,
            (1, 1,) + args["data_dim"],
            bias_rank=fc_bias,
            feature_dims=feature_dims,
            **args_out
        )

    def forward(
        self, past_input: torch.Tensor, future_input: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Forward step for the generator.

        :param past_input: input data from past
        :param future_input: input data from future
        :param noise: input noise
        :return: estimation of MMD score
        """
        past_input = self.relu(self.emb_layer(past_input))
        future_input = self.relu(self.emb_layer(future_input))

        _, h_t = self.rnn_enc_layer(past_input)

        # right shifting of future data on one step
        future_inp_shift = future_input.clone()
        future_inp_shift[0].data.fill_(0)
        future_inp_shift[1:] = future_input[:-1]

        hidden = h_t + noise
        future_target, _ = self.rnn_dec_layer(future_inp_shift, hidden)

        output = self.output_layer(future_target)

        return output


class KlcpdDiscTl(nn.Module):
    """Class for KL-CPD discriminator with tensor-layer decomposition."""

    def __init__(self, args) -> None:
        """Initialize discriminator with tensor-layer decomposition.

        :param args: dict with the network parameters:
            - type of tensor-layer decomposition
            - data dimensions
            - embedding dimensions
            - hidden dimension for RNN
            - bias rank for tensor-layer
        """
        super().__init__()
        block_type = args["block_type"]
        self.rnn_hid_dims = args["rnn_hid_dim"]
        self.emb_dims = args["emb_dim"]
        self.relu = nn.ReLU()
        self.data_dim = args["data_dim"]
        feature_dims = args["feature_dim"] 

        fc_bias, gru_bias = init_bias(args)
        block, (args_in, args_gru, args_out) = init_fc_rnn_tl(block_type, args)

        self.emb_layer = block(
            (1, 1,) + self.data_dim,
            (1, 1,) + self.emb_dims,
            bias_rank=fc_bias,
            feature_dims=feature_dims,
            **args_in
        )

        self.rnn_enc_layer = GruTl(
            block_type,
            (1,) + self.emb_dims,
            (1,) + self.rnn_hid_dims,
            bias_rank=gru_bias,
            feature_dims=feature_dims,
            **args_gru
        )

        self.rnn_dec_layer = GruTl(
            block_type,
            (1,) + self.rnn_hid_dims,
            (1,) + self.emb_dims,
            bias_rank=gru_bias,
            feature_dims=feature_dims,
            **args_gru
        )

        self.output_layer = block(
            (1, 1,) + self.emb_dims,
            (1, 1,) + self.data_dim,
            bias_rank=fc_bias,
            feature_dims=feature_dims,
            **args_out
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward step for the discriminator.

        :param input: input data
        :return: encoded and decoded inputs
        """
        if len(input.shape) == 3:
            # X is flatten
            input = input.reshape(*input.shape[:2], *self.data_dim)

        output = self.relu(self.emb_layer(input))

        output_enc, _ = self.rnn_enc_layer(output)
        output_dec, _ = self.rnn_dec_layer(output_enc)

        output_dec = self.relu(self.output_layer(output_dec))
        return output_enc, output_dec


#################################BCE MODEL############################################
class BceRNN(nn.Module):
    """Class for core BCE model with tensor-layer decomposition."""

    def __init__(self, args: dict) -> None:
        """Initialize BCE model.

        :param args: dict with the network parameters:
            - data dimensions
            - embedding dimensions
            - hidden dimension for RNN
        """
        super().__init__()

        self.rnn_hid_dims = args["rnn_hid_dim"]
        self.relu = nn.ReLU()
        self.data_dim = args["data_dim"]

        self.rnn = nn.LSTM(
            self.data_dim,
            self.rnn_hid_dims,
            num_layers=args["num_layers"],
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.rnn_hid_dims, 1)

    def forward(self, input: torch.Tensor) -> object:
        """Forward step of BCE model.

        :param input: input data
        :return: predictions of change points
        """
        output = input.flatten(2)
        output, _ = self.rnn(output)
        output = self.output_layer(output)  # batch_size, timesteps, 1
        output = torch.sigmoid(output)
        return output


class BceRNNTl(nn.Module):
    """Class for core BCE model with tensor-layer decomposition."""

    def __init__(self, args) -> None:
        """Initialize BCE model with tensor-layer decomposition.

        :param args: dict with the network parameters:
            - type of tensor-layer decomposition
            - data dimensions
            - embedding dimensions
            - hidden dimension for RNN
            - bias rank for tensor-layer
        """
        super().__init__()
        block_type = args["block_type"]
        self.rnn_hid_dims = args["rnn_hid_dim"]
        self.emb_dims = args["emb_dim"]
        self.relu = nn.ReLU()
        self.data_dim = args["data_dim"]
        feature_dims = args["feature_dim"] 

        fc_bias, rnn_bias = init_bias(args)
        block, (args_in, args_rnn, args_out) = init_fc_rnn_tl(block_type, args)

        self.rnn = GruTl(
            block_type,
            (1,) + self.data_dim,
            (1,) + self.rnn_hid_dims,
            bias_rank=rnn_bias,
            feature_dims=feature_dims,
            batch_first=True,
            **args_rnn
        )

        self.output_layer = block(
            (1, 1,) + self.rnn_hid_dims,
            (1, 1,) + (1, 1, 1),
            bias_rank=fc_bias,
            feature_dims=feature_dims,
            normalize="in",
            **args_out
        )

    def forward(self, input):
        """Forward step of BCE model.

        :param input: input data
        :return: predictions of change points
        """
        output, _ = self.rnn(input)
        output = self.output_layer(output)
        output = output.reshape(*output.shape[:2], 1)  # flatten across last 3 dim
        output = torch.sigmoid(output)
        return output


class BceRNNTl_v2(nn.Module):
    def __init__(self, args: Dict) -> None:
        super().__init__()
        self.input_layer, self.rnn, self.output_layer = parse_bce_args(args)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        output = self.input_layer(input)
        
        output, _ = self.rnn(output)
        output = self.output_layer(output)
        output = output.reshape(*output.shape[:2], 1)  # flatten across last 3 dim
        output = torch.sigmoid(output)
        return output