import torch
import torch.nn as nn
import torch.nn.functional as F

class MAML(nn.Module):
    def __init__(self):
        super(MAML, self).__init__()

    def forward(self, x, params):
        x = F.relu(F.linear(x, params['input_net_weight'], params['input_net_bias']))
        x = F.relu(F.linear(x, params['latent_net_weight'], params['latent_net_bias']))
        x = F.linear(x, params['output_net_weight'], params['output_net_bias'])
        return x
