import torch.nn.functional as F
import click
import gridfs
import time
import torch
import numpy as np
from const import *
from pymongo import MongoClient
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.vae import VAE, ConvVAE
from models.helper import load_model, save_checkpoint
from lib.dataset import VAEDataset
from lib.visu import create_img_recons, traverse_latent_space
from lib.train_utils import create_optimizer, fetch_new_run, create_state


def loss_fn(recon_x, x, mu, logvar):
    """
    Loss function of β-VAE, check https://arxiv.org/pdf/1804.03599.pdf
    or  https://dylandjian.github.io/world-models/
    """

    batch_size = x.size()[0]
    if VAE_LOSS == "bce":
        loss = F.binary_cross_entropy(recon_x, x, size_average=False)
    else:
        loss = F.mse_loss(recon_x, x, size_average=False)

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss /= batch_size
    kld /= batch_size
    return loss + BETA * kld.sum()


def train_epoch(vae, optimizer, frames):
    """ Train the VAE over a batch of example frames """

    optimizer.zero_grad()

    recon_x, mu, logvar = vae(frames)
    loss = loss_fn(recon_x, frames, mu, logvar)

    loss.backward()
    optimizer.step()

    return float(loss)


def train_vae(current_time):
    """
    Train a VAE to create a latent representation of the Sonic levels by
    trying to encode and decode each frame.
    """

    dataset = VAEDataset()
    last_id = 0
    lr = LR
    version = 1
    total_ite = 1

    client = MongoClient()
    db = client.retro_contest
    collection = db[current_time]
    fs = gridfs.GridFS(db)

    ## Load or create models
    vae, checkpoint = load_model(current_time, -1, model="vae")
    if not vae:
        vae = ConvVAE((WIDTH, HEIGHT, 3), LATENT_VEC).to(DEVICE)
        optimizer = create_optimizer(vae, lr)
        state = create_state(version, lr, total_ite, optimizer)
        save_checkpoint(vae, "vae", state, current_time)
    else:
        optimizer = create_optimizer(vae, lr, param=checkpoint['optimizer'])
        total_ite = checkpoint['total_ite'] + 1
        lr = checkpoint['lr']
        version = checkpoint['version']
        last_id = 0

    ## Fill the dataset (or wait for the database to be filled)
    while len(dataset) < SIZE:
        last_id = fetch_new_run(collection, fs, dataset, last_id, loaded_version=current_time)
        time.sleep(5)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_VAE, shuffle=True)
    while True:
        batch_loss = []
        running_loss = []

        for batch_idx, frames in enumerate(dataloader):
            frames = torch.tensor(frames, dtype=torch.float, device=DEVICE) / 255
            frames = frames.view(-1, 3, WIDTH, HEIGHT)

            ## Save the models
            if total_ite % SAVE_TICK == 0:
                version += 1
                state = create_state(version, lr, total_ite, optimizer)
                save_checkpoint(vae, "vae", state, current_time)
            
            if total_ite % SAVE_PIC_TICK == 0:
                traverse_latent_space(vae, frames[0], frames[-1], total_ite)
                create_img_recons(vae, frames[0:40], total_ite)

            loss = train_epoch(vae, optimizer, frames)
            running_loss.append(loss)

            ## Print running loss
            if total_ite % LOSS_TICK == 0:
                print("[TRAIN] current iteration: %d, averaged loss: %.3f"\
                        % (total_ite, loss))
                batch_loss.append(np.mean(running_loss))
                running_loss = []
            
            ## Fetch new games
            if total_ite % REFRESH_TICK == 0:
                new_last_id = fetch_new_run(collection, fs, dataset, last_id)
                if new_last_id == last_id:
                    last_id = 0
                else:
                    last_id = new_last_id
            
            total_ite += 1
    
        if len(batch_loss) > 0:
            print("[TRAIN] Average backward pass loss : %.3f, current lr: %f" % (np.mean(batch_loss), lr))
    

@click.command()
@click.option("--folder", default=-1)
def main(folder):
    if folder == -1:
        current_time = int(time.time())
    else:
        current_time = folder
    train_vae(str(current_time))


if __name__ == "__main__":
    main()