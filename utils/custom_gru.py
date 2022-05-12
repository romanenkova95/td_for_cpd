import torch.nn as nn
from models_v2 import TCL

class custom_GRU_cell(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # update gate
        self.linear_w_z = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_u_z = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activation_z = nn.Sigmoid()

        # reset gate
        self.linear_w_r = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_u_r = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activation_r = nn.Sigmoid()
        
        # output
        self.linear_w_h = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_u_h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.activation_h = nn.Tanh()
        
        
    def forward(self, x_t, h_prev):
        try:
            device = x_t.device
        except:
            device = 'cpu'
                
        output_z = self.activation_z(self.linear_w_z(x_t) + self.linear_u_z(h_prev))
        output_r = self.activation_r(self.linear_w_r(x_t) + self.linear_u_r(h_prev))
        hidden_hat = self.activation_h(self.linear_w_h(x_t) + torch.mul(output_r, self.linear_u_h(h_prev)))
        
        ones = torch.ones_like(output_z).to(device)
        hidden = torch.mul(output_z, h_prev) + torch.mul((ones - output_z), hidden_hat)
        
        return hidden
    
    
class custom_GRU(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.cell = custom_GRU_cell(input_dim, hidden_dim)
    
    def forward(self, inputs):
        try:
            device = inputs.device
        except:
            device = 'cpu'

        outputs = []
        out_t = torch.zeros(inputs.shape[0], 1, self.hidden_dim)
        
        for t, x_t in enumerate(inputs.chunk(inputs.shape[1], dim=1)):
            out_t = self.cell(x_t, out_t)
            outputs.append(out_t.squeeze(1).detach().cpu())
        outputs = torch.stack(outputs, 1)
        return outputs, out_t.squeeze(1)



class custom_GRU_cell_TCL(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear_w = nn.Linear(self.input_dim, self.hidden_dim*3)
        self.linear_u = nn.Linear(self.hidden_dim, self.hidden_dim*3, bias=False)
        self.activation_zr = nn.Sigmoid()
        self.activation_h = nn.Tanh()
        
        
    def forward(self, x_t, h_prev):

        # x_t shape: N, Hin; h_prev shape: 
        try:
            device = x_t.device
        except:
            device = 'cpu'
                
        x_zrh = self.linear_w(x_t).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        x_z, x_r, x_h = x_zrh
        output_z = self.activation_z(self.linear_w_z(x_t) + self.linear_u_z(h_prev))
        output_r = self.activation_r(self.linear_w_r(x_t) + self.linear_u_r(h_prev))
        hidden_hat = self.activation_h(self.linear_w_h(x_t) + torch.mul(output_r, self.linear_u_h(h_prev)))
        
        ones = torch.ones_like(output_z).to(device)
        hidden = torch.mul(output_z, h_prev) + torch.mul((ones - output_z), hidden_hat)
        
        return hidden
    
    
class custom_GRU_TCL(torch.nn.Module):

    def __init__(self, input_dims, hidden_dims, ranks_u, ranks_h):
        super().__init__()
        
        self.cell = custom_GRU_cell_TCL(input_dims, hidden_dims, ranks_u, ranks_h)
    
    def forward(self, inputs):

        # inputs shape: L, N, Hin
        L, N, *input_dims = inputs.shape

        outputs = []
        out_t = torch.zeros(N, self.hidden_dim).to(inputs)
        
        for x_t in inputs:
            out_t = self.cell(x_t, out_t)
            outputs.append(out_t.squeeze(1).detach().cpu()) # FIXME check squeeze dim
        outputs = torch.stack(outputs, 1)
        return outputs, out_t.squeeze(1)