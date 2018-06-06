import torch.nn.functional as F
import click
import gridfs
import time
import torch
import numpy as np
from const import *
from models.helper import load_model, save_checkpoint
from pymongo import MongoClient
from models.vae import VAE, ConvVAE
from torch.utils.data import DataLoader
from lib.dataset import FrameDataset
from lib.visu import create_img, create_img_recons
from torchvision.utils import save_image
from lib.train_utils import create_optimizer, fetch_new_run, create_state


def loss_fn(recon_x, x, mu, logvar):
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
    """ Used to train the 3 models over a single batch """

    optimizer.zero_grad()
    recon_x, mu, logvar = vae(frames)
    loss = loss_fn(recon_x, frames, mu, logvar)
    loss.backward()
    optimizer.step()

    return float(loss)


def train_vae(current_time):
    dataset = FrameDataset()
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
    
    while len(dataset) < SIZE:
        last_id = fetch_new_run(collection, fs, dataset, last_id, loaded_version=current_time)
        time.sleep(5)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    while True:
        batch_loss = []
        for batch_idx, (frames) in enumerate(dataloader):
            running_loss = []
            # lr, optimizer = update_lr(lr, optimizer, total_ite)

            ## Save the models
            if total_ite % SAVE_TICK == 0:
                version += 1
                state = create_state(version, lr, total_ite, optimizer)
                save_checkpoint(vae, "vae", state, current_time)
            
            if total_ite % SAVE_PIC_TICK == 0:
                create_img_recons(vae, encoded_frames, version)
    
            encoded_frames = torch.tensor(frames, dtype=torch.float, device=DEVICE).div(255)
            loss = train_epoch(vae, optimizer, encoded_frames)
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