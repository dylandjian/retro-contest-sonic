import torch.nn as nn
import torch
import torch.nn.functional as F
from const import *


class Controller(nn.Module):
    def __init__(self, hidden_dim, hidden_units, action_space):
        super(Controller, self).__init__()

        self.fc1 = nn.Linear(hidden_dim + hidden_units, action_space)
        
    def forward(self, x):
        prob = F.softmax(self.fc1(x), dim=1)
        return prob


