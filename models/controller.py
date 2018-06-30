import torch.nn as nn
import torch
import torch.nn.functional as F
from const import *


class Controller(nn.Module):
    def __init__(self, hidden_dim, action_space):
        super(Controller, self).__init__()

        self.action_space = action_space
        self.fc1 = nn.Linear(hidden_dim, action_space)
        
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        return x
    
