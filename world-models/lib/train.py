import torch.nn.functional as F
import torch





def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD



def train(dataset, vae):
    while True:
        optimizer = optim.Adam(vae.parameters(), lr=LR)
