import torch
import torch.nn as nn

from typing import Tuple

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
        X_p = X_p.flatten(2) # batch_size, timesteps, C*H*W
        X_f = X_f.flatten(2) # batch_size, timesteps, C*H*W

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

        X = X.flatten(2)
        X = self.relu(self.fc1(X))
        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)
        X_dec = self.relu(self.fc2(X_dec))
        
        return X_enc, X_dec