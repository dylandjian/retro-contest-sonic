import torch.nn.functional as F
import matplotlib.pyplot as plt
import gridfs
import time
import torch
import numpy as np
import random
import click
from const import *
from models.helper import load_model, save_checkpoint
from pymongo import MongoClient
from lib.dataset import FrameDataset
from models.lstm import LSTM
from models.vae import ConvVAE
from torch.utils.data import DataLoader
from torch.distributions import Normal
from lib.train_utils import create_optimizer, fetch_new_run, create_state
from torchvision.utils import save_image


def mdn_loss_function(out_pi, out_sigma, out_mu, y):
    """
    Mixed Density Network loss function, see : 
    https://mikedusenberry.com/mixture-density-networks
    """

    result = Normal(loc=out_mu, scale=out_sigma)
    y = y.view(-1, SEQUENCE, LATENT_VEC)
    result = torch.exp(result.log_prob(y))
    result = torch.sum(result * out_pi, dim=2)
    result = -torch.log(EPSILON + result)
    return torch.mean(result)


def train_epoch(lstm, optimizer, example):
    """ Used to train the 3 models over a single batch """

    optimizer.zero_grad()

    ## Concatenate action to encoded vector
    x = torch.cat((example['encoded'][:-OFFSET],
            example['actions'][:-OFFSET].view(-1, 1)), dim=1)
    x = x.view(-1, SEQUENCE, LATENT_VEC + 1)
    ## Shift target encoded vector
    target = example['encoded'][OFFSET:example['encoded'].size()[0]]

    pi, sigma, mu = lstm(x)
    loss = mdn_loss_function(pi, sigma, mu, target)
    loss.backward()
    optimizer.step()

    return float(loss)


def sample(pi, mu, sigma):
    sampled = torch.sum(pi * torch.normal(mu, sigma), dim=2)
    return sampled.view(1, LATENT_VEC)


def sample_long_term(vae, lstm, frames, version):
    """ Given a frame, tries to predict the next 100 encoded vectors """

    with torch.no_grad():
        frames_z = vae(frames.view(-1, 3, WIDTH, HEIGHT)[0:4], encode=True)
        first_z = frames_z[0].view(1, LATENT_VEC)
        for i in range(1, 20):
            if i == 1:
                new_state = torch.cat((first_z, torch.full((1, 1), 1, device=DEVICE) \
                                .div(ACTION_SPACE)), dim=1)
            else:
                new_state = torch.cat((z, torch.full((1, 1), 1, device=DEVICE)\
                                .div(ACTION_SPACE)), dim=1)
            pi, sigma, mu = lstm(new_state.view(1, 1, LATENT_VEC + 1))
            z = sample(pi, mu, sigma)
            if i == 1:
                res = torch.cat((frames[0:4], vae.decode(frames_z).view(-1, 3, WIDTH, HEIGHT), 
                        vae.decode(z)[0].view(1, 3, WIDTH, HEIGHT)))
            else:
                res = torch.cat((res, vae.decode(z)[0].view(1, 3, WIDTH, HEIGHT)))
        save_image(res, "results/test-{}.png".format(version))


def init_models(current_time):
    """ Init variables for code clarity despite the long return """

    last_id = 0
    lr = LR
    version = 1
    total_ite = 1

    vae, _ = load_model(current_time, -1, model="vae")
    if not vae:
        vae = ConvVAE((HEIGHT, WIDTH, 3), LATENT_VEC).to(DEVICE)

    lstm, checkpoint = load_model(current_time, -1, model="lstm")
    if not lstm:
        lstm = LSTM(HIDDEN_UNITS, LATENT_VEC,\
                     NUM_LAYERS, GAUSSIANS, HIDDEN_DIM).to(DEVICE)
        optimizer = create_optimizer(lstm, lr)
        state = create_state(version, lr, total_ite, optimizer)
        save_checkpoint(lstm, "lstm", state, current_time)
    else:
        optimizer = create_optimizer(lstm, lr, param=checkpoint['optimizer'])
        total_ite = checkpoint['total_ite']
        lr = checkpoint['lr']
        version = checkpoint['version']
        last_id = 0

    return vae, lstm, optimizer, total_ite, lr, version, last_id



def train_lstm(current_time):
    dataset = FrameDataset(lstm=True)
    client = MongoClient()
    db = client.retro_contest
    collection = db[current_time]
    fs = gridfs.GridFS(db)

    vae, lstm, optimizer, total_ite, lr, version, last_id = init_models(current_time)
    while len(dataset) < SIZE:
        last_id = fetch_new_run(collection, fs, dataset, last_id, loaded_version=current_time)
        time.sleep(5)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    while True:
        lstm.hidden = lstm.init_hidden()
        batch_loss = []
        for batch_idx, (frames, actions) in enumerate(dataloader):
            running_loss = []
            # lr, optimizer = update_lr(lr, optimizer, total_ite)

            ## Save the model
            if total_ite % SAVE_TICK == 0:
                version += 1
                state = create_state(version, lr, total_ite, optimizer)
                save_checkpoint(lstm, "lstm", state, current_time)

            lstm.hidden = [data.detach() for data in lstm.hidden]
            frames = torch.tensor(frames, dtype=torch.float, device=DEVICE).div(255)

            ## Save a picture of the long term sampling
            if total_ite % SAVE_PIC_TICK == 0:
                sample_long_term(vae, lstm, frames, version)

            encoded = vae(frames, encode=True)
            example = {
                'encoded': encoded,
                'actions' : torch.tensor(actions, dtype=torch.float, device=DEVICE).div(ACTION_SPACE)
            }

            loss = train_epoch(lstm, optimizer, example)
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
    train_lstm(str(current_time))


if __name__ == "__main__":
    main()
