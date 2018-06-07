import torch.nn as nn
import torch
import torch.nn.functional as F
from const import *


class Controller(nn.Module):
    def __init__(self, hidden_dim, hidden_units, action_space):
        super(Controller, self).__init__()

        self.action_space = action_space
        self.fc1 = nn.Linear(hidden_dim + hidden_units, 512)
        self.fc2 = nn.Linear(512, action_space)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x
    
