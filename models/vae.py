import torch.nn as nn
from const import *
import torch
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, input_shape, z_dim):
        super(ConvVAE, self).__init__()
        
        self.image_size = 3 * WIDTH * HEIGHT
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)

        self.fc1 = nn.Linear(256 * 6 * 6, z_dim)
        self.fc2 = nn.Linear(256 * 6 * 6, z_dim)
        self.fc3 = nn.Linear(z_dim, 256 * 6 * 6)

        self.deconv1 = nn.ConvTranspose2d(256 * 6 * 6, 128, 5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 5, stride=2)
        self.deconv4 = nn.ConvTranspose2d(32, 16, 6, stride=2)
        self.deconv5 = nn.ConvTranspose2d(16, 3, 6, stride=2)


    def encode(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = h.view(-1, 256 * 6 * 6)
        return self.fc1(h), self.fc2(h)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu


    def decode(self, z):
        h = self.fc3(z).view(-1, 256 * 6 * 6, 1, 1)
        h = F.relu(self.deconv1(h))
        h = F.relu(self.deconv2(h))
        h = F.relu(self.deconv3(h))
        h = F.relu(self.deconv4(h))
        h = F.sigmoid(self.deconv5(h))
        return h


    def forward(self, x, encode=False, mean=False):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if encode:
            if mean:
                return mu
            return z
        return self.decode(z), mu, logvar


class VAE(nn.Module):
    def __init__(self, input_shape, hidden_dim, z_dim):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.image_size = 3 * WIDTH * HEIGHT
        self.fc1 = nn.Linear(self.image_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(hidden_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, self.image_size)


    def encode(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)


    def reparameterize(self, mu, logvar, training=False):
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps* std + mu
        else:
            return mu


    def decode(self, z):
        hidden = self.relu(self.fc4(z))
        return F.sigmoid(self.fc5(hidden))


    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.image_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
