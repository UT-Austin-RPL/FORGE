import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class ConvGRUCell_3D(nn.Module):
    """
    Generate a convolutional GRU cell
    """
    def __init__(self, config, input_size, hidden_size):
    
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.conv_gate = nn.Conv3d(self.input_size + self.hidden_size, self.hidden_size*2, 3, padding=1)
        self.out_gate = nn.Conv3d(self.input_size + self.hidden_size, self.hidden_size, 3, padding=1)

    def forward(self, x, prev_state=None):

        b, c, d, h, w = x.shape

        if prev_state is None:
            state_size = [b, self.hidden_size, d, h, w]
            prev_state = torch.zeros(state_size).to(x.device)

        x_conv = self.conv_gate(torch.cat([x, prev_state], dim=1))
        update, reset = torch.split(x_conv, self.hidden_size, dim=1)
        update, reset = torch.sigmoid(update), torch.sigmoid(reset)

        out_inputs = torch.tanh(self.out_gate(torch.cat([x, prev_state * reset], dim=1)))

        return prev_state * (1 - update) + out_inputs * update



class ConvGRU_3D(nn.Module):
    def __init__(self, config, n_layers=1, input_size=16, hidden_size=16):
        '''
        Generates a multi-layer 3D convolutional GRU.
        input/hidden size is the number of channels
        '''
        super(ConvGRU_3D, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.n_layers = n_layers

        cells = []
        for i in range(self.n_layers):
            input_dim = self.input_size if i == 0 else hidden_size
            cell = ConvGRUCell_3D(config, input_dim, self.hidden_size)
            cells.append(cell)
        
        self.cells = nn.ModuleList(cells)

        self.fusion_norm = nn.BatchNorm3d(hidden_size)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(input_size, input_size, 3, padding=1),
            nn.BatchNorm3d(input_size),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(input_size, input_size, 3, padding=1),
            nn.BatchNorm3d(input_size),
            nn.LeakyReLU(inplace=True),
        )


    def forward(self, x, hidden=None):
        '''
        x: [b,t,c,d,h,w]
        '''
        seq_len = x.shape[1]

        if not hidden:
            hidden = [None]*self.n_layers

        layer_output_list = []
        last_state_list = []

        cur_layer_input = x

        for layer_idx in range(self.n_layers):
            h = hidden[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cells[layer_idx](cur_layer_input[:, t, :, :, :, :], h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)  # [b,t,c',d,h,w]
            cur_layer_input = layer_output

        return self.fusion_norm(h)

