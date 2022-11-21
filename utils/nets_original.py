import torch
import torch.nn as nn

from typing import Tuple
from utils import models_v2 as models

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


class NetG_Masked(nn.Module):
    def __init__(self, args) -> None:
            
        super().__init__()
        self.RNN_hid_dim = args['RNN_hid_dim']
        self.emb_dim = args['emb_dim']
        self.relu = nn.ReLU()
        
        self.fc = nn.Linear(args['data_dim'], self.emb_dim)
        self.rnn_enc_layer = nn.GRU(self.emb_dim, self.RNN_hid_dim, num_layers=args['num_layers'], batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.emb_dim, self.RNN_hid_dim, num_layers=args['num_layers'], batch_first=True)
        self.fc_layer = nn.Linear(self.RNN_hid_dim, args['data_dim'])

        # self.masks = nn.ParameterList([
        #     nn.Parameter(torch.ones(1, 1, args['data_dim'])),
        #     nn.Parameter(torch.ones(1, 1, args['emb_dim'])),
        #     nn.Parameter(torch.ones(1, 1, args['data_dim'])),
        # ])

        # self.clip = lambda x: torch.clip(x, 0, 1)
        # self.clip = lambda x: x

  
    def forward(self, X_p, X_f, noise) -> torch.Tensor:

        # self.masks = [mask_i.to(X_p) for mask_i in self.masks]
        # self.masks = self.masks.to(X_p)

        X_p = X_p.flatten(2)# * self.clip(self.masks[0]) # batch_size, timesteps, C*H*W
        X_p = self.fc(X_p)# * self.clip(self.masks[1])
        X_p = self.relu(X_p)

        X_f = X_f.flatten(2)# * self.clip(self.masks[0]) # batch_size, timesteps, C*H*W
        X_f = self.fc(X_f)# * self.clip(self.masks[1])
        X_f = self.relu(X_f)

        X_p_enc, h_t = self.rnn_enc_layer(X_p)
        X_f_shft = self.shft_right_one(X_f)
        hidden = h_t + noise
        Y_f, _ = self.rnn_dec_layer(X_f_shft, hidden)
        output = self.fc_layer(Y_f)# * self.clip(self.masks[2])
        return output

    def shft_right_one(self, X) -> torch.Tensor:
        X_shft = X.clone()
        X_shft[:, 0, :].data.fill_(0)
        X_shft[:, 1:, :] = X[:, :-1, :]
        return X_shft
    
    # def mask_loss(self, p=1):
    #     return sum([torch.norm(mask_i, p) for mask_i in self.masks])    

    # def mask_loss(self, p=1):

    #     return self.fc.weight.norm(p=1) + self.fc.bias.norm(p=1) \
    #          + self.fc_layer.weight.norm(p=1) + self.fc_layer.bias.norm(p=1)


class NetD_Masked(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.RNN_hid_dim = args['RNN_hid_dim']
        self.emb_dim = args['emb_dim']
        
        self.fc1 = nn.Linear(args['data_dim'], self.emb_dim)

        self.rnn_enc_layer = nn.GRU(self.emb_dim, self.RNN_hid_dim, num_layers=args['num_layers'], batch_first=True)
        self.rnn_dec_layer = nn.GRU(self.RNN_hid_dim, self.emb_dim, num_layers=args['num_layers'], batch_first=True)

        self.fc2 = nn.Linear(self.emb_dim, args['data_dim'])
        self.relu = nn.ReLU()

        self.mask1 = torch.ones(1, 1, args['emb_dim'])
        self.mask2 = torch.ones(1, 1, args['emb_dim'])
        
    def forward(self, X) -> Tuple[torch.Tensor, torch.Tensor]:

        self.mask1 = self.mask1.to(X)
        self.mask2 = self.mask2.to(X)

        X = X.flatten(2)
        X = self.fc1(X) * self.mask1
        X = self.relu(X)

        X_enc, _ = self.rnn_enc_layer(X)
        X_dec, _ = self.rnn_dec_layer(X_enc)

        X_dec = X_dec * self.mask2
        X_dec = self.fc2(X_dec)
        X_dec = self.relu(X_dec)
        
        return X_enc, X_dec

    def mask_loss(self, p=1):

        return self.fc1.weight.norm(p=1) + self.fc1.bias.norm(p=1) \
             + self.fc2.weight.norm(p=1) + self.fc2.bias.norm(p=1)
    
    
##############################################################################################
#############################BCE MODEL##################################
class BCE_GRU(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
        self.RNN_hid_dims = args['RNN_hid_dim']
        self.emb_dims = args['emb_dim']
        self.relu = nn.LeakyReLU(0.1)
        self.data_dim = args['data_dim']

        #self.fc_1 = nn.Linear(self.data_dim, self.emb_dims)
        self.rnn = nn.GRU(self.data_dim, self.RNN_hid_dims, num_layers=args['num_layers'], batch_first=True)        
        self.fc_2 = nn.Linear(self.RNN_hid_dims, 1)
            
        
    def forward(self, x):
        x = x.flatten(2)
        #x = self.relu(self.fc_1(x)) # batch_size, timesteps, emb_dim
        x, _ = self.rnn(x)
        x = self.fc_2(x) # batch_size, timesteps, 1  
        x = x.reshape(*x.shape[:2], 1)
        x = torch.sigmoid(x)
        print(x)
        return x