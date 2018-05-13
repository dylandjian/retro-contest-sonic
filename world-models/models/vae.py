import torch.nn as nn
import torch
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_shape, hidden_dim, z_dim):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.image_size = input_shape[0] * input_shape[1] * input_shape[2]
        self.fc1 = nn.Linear(self.image_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(hidden_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, self.image_size)


    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu


    def decode(self, z):
        hidden = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(hidden))


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.image_size))
        z = self.reparameterize(mu, logvar)
        x_reconst = self.decode(z).view(x.size())
        return x_reconst, mu, logvar
