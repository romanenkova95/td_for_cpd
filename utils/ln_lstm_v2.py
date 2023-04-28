import torch
import torch.nn as nn 

class LayerNormLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first=True, ln_type='after'):
        super().__init__()
        
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.batch_first = batch_first

        self.layer_norm_w = nn.Identity()
        self.layer_norm_u = nn.Identity()  
        if ln_type == 'before':
            #print("Add LayerNorm BEFORE each Linear layer")
            self.linear_w = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.input_dim), nn.Linear(self.input_dim, self.hidden_dim)) 
                                           for _ in range(4)])
            self.linear_u = nn.ModuleList([nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.hidden_dim))
                                           for _ in range(4)])
        elif ln_type == "after":
            #print("Add LayerNorm AFTER each Linear layer")
            self.linear_w = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim)) 
                                           for _ in range(4)])
            self.linear_u = nn.ModuleList([nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim))
                                           for _ in range(4)])
        elif ln_type is None:
            self.linear_w = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim) for _ in range(4)])
            self.linear_u = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(4)])
            self.layer_norm_w = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(4)])
            self.layer_norm_u = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(4)])
        
    def forward(self, inputs, init_states=None):

        if self.batch_first:
            # sequence first (timesteps, batch_size, input_dims)
            inputs = inputs.transpose(0, 1)
        N = inputs.shape[1]

        outputs = []
        if init_states is None:
            h_prev = torch.zeros(N, self.hidden_dim).to(inputs)
            c_prev = torch.zeros(N, self.hidden_dim).to(inputs)
        else:
            h_prev, c_prev = init_states

        outputs = []
        for i, x_t in enumerate(inputs):
            x_f, x_i, x_o, x_c_hat = [linear(x_t) for linear in self.linear_w]
            h_f, h_i, h_o, h_c_hat = [linear(h_prev) for linear in self.linear_u]

            #if i == len(inputs) - 1:
            #    x_f = self.layer_norm_w[0](x_f)
            #    x_i = self.layer_norm_w[1](x_i)
            #    x_o = self.layer_norm_w[2](x_o)
            #    x_c_hat = self.layer_norm_w[3](x_c_hat)

            #    h_f = self.layer_norm_u[0](h_f)
            #    h_i = self.layer_norm_u[1](h_i)
            #    h_o = self.layer_norm_u[2](h_o)
            #    h_c_hat = self.layer_norm_u[3](h_c_hat)

            f_t = torch.sigmoid(x_f + h_f)
            i_t = torch.sigmoid(x_i + h_i)
            o_t = torch.sigmoid(x_o + h_o)
            c_t_hat = torch.tanh(x_c_hat + h_c_hat)
            c_prev = torch.mul(f_t, c_prev) + torch.mul(i_t, c_t_hat)
            h_prev = torch.mul(o_t, torch.tanh(c_prev))
            outputs.append(h_prev)

        outputs = torch.stack(outputs, dim=0)
        if self.batch_first:
            # batch first (batch_size, timesteps, input_dims)
            outputs = outputs.transpose(0, 1)

        return outputs, (h_prev, c_prev)
    

class LayerNormTemporal(nn.Module):
    def __init__(self, normalized_shape, transpose=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape)
        self.transpose = transpose

    def forward(self, inputs):
        if self.transpose:
            inputs = inputs.transpose(1, 2)
        outputs = self.norm(inputs)
        if self.transpose:
            outputs = outputs.transpose(1, 2)
        return outputs


