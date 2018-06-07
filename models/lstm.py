import torch.nn as nn
import torch
import torch.nn.functional as F
from const import *


class LSTM(nn.Module):
    def __init__(self, sequence_len, hidden_units, z_dim, num_layers, n_gaussians, hidden_dim):
        super(LSTM, self).__init__()

        self.n_gaussians = n_gaussians
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.hidden_units = hidden_units
        self.sequence_len = sequence_len
        self.hidden = self.init_hidden(self.sequence_len)

        self.fc1 = nn.Linear(self.z_dim + 1, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, hidden_units, num_layers)
        self.z_pi = nn.Linear(hidden_units, n_gaussians * self.z_dim)
        self.z_sigma = nn.Linear(hidden_units, n_gaussians * self.z_dim)
        self.z_mu = nn.Linear(hidden_units, n_gaussians * self.z_dim) 


    def forward(self, x):
        self.lstm.flatten_parameters()
        sequence = x.size()[1]

        ## Hidden state
        x = F.relu(self.fc1(x))
        z, self.hidden = self.lstm(x, self.hidden)

        pi = self.z_pi(z).view(-1, sequence, self.n_gaussians, self.z_dim)
        pi = F.softmax(pi, dim=2)
        pi = pi / TEMPERATURE

        sigma = torch.exp(self.z_sigma(z)).view(-1, sequence,
                        self.n_gaussians, self.z_dim)
        sigma = sigma * (TEMPERATURE ** 0.5)
        mu = self.z_mu(z).view(-1, sequence, self.n_gaussians, self.z_dim)
    
        return pi, sigma, mu


    def init_hidden(self, sequence):
        hidden = torch.zeros(self.num_layers, sequence, self.hidden_units, device=DEVICE)
        cell = torch.zeros(self.num_layers, sequence, self.hidden_units, device=DEVICE)
        return hidden, cell


