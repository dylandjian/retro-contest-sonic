import torch.nn as nn
import torch
import torch.nn.functional as F
from const import *


class Controller(nn.Module):
    def __init__(self, hidden_dim, hidden_units, action_space):
        super(Controller, self).__init__()

        self.action_space = action_space
        self.fc1 = nn.Linear(hidden_dim + hidden_units, 1)
        
    def forward(self, x):
        res = ((F.tanh(self.fc1(x)) + 1) * (self.action_space)) / 2
        return torch.trunc(res)
    
