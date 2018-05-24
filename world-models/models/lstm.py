import torch.nn as nn
import torch
import torch.nn.functional as F
from const import *


class LSTM(nn.Module):
    def __init__(self, hidden_units, z_dim, num_layers, n_gaussians, hidden_dim):
        super(LSTM, self).__init__()

        self.n_gaussians = n_gaussians
        self.num_layers = num_layers
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.hidden_units = hidden_units
        self.hidden = self.init_hidden()

        # self.fc1 = nn.Linear(self.z_dim + 1, hidden_dim)
        self.lstm = nn.LSTM(self.z_dim + 1, hidden_units, num_layers)
        self.z_pi = nn.Linear(hidden_units, n_gaussians)
        self.z_sigma = nn.Linear(hidden_units, n_gaussians)
        self.z_mu = nn.Linear(hidden_units, n_gaussians) 


    def forward(self, x):
        ## Hidden state
        # z = F.tanh(self.fc1(x))
        z, self.hidden = self.lstm(x, self.hidden)

        ## Probabilities for each Gaussian
        pi = self.z_pi(z).view(-1, self.n_gaussians, 1)
        pi = F.softmax(pi, dim=1)

        sigma = torch.exp(self.z_sigma(z)).view(-1, self.n_gaussians, 1)
        mu = self.z_mu(z).view(-1, self.n_gaussians, 1)
        return pi, sigma, mu


    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_units, device=DEVICE)
        cell = torch.zeros(self.num_layers, 1, self.hidden_units, device=DEVICE)
        return hidden, cell


