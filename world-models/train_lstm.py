import torch.nn.functional as F
import matplotlib.pyplot as plt
import gridfs
import time
import torch
import numpy as np
import click
from const import *
from models.helper import load_model, save_checkpoint
from pymongo import MongoClient
from lib.dataset import FrameDataset
from models.lstm import LSTM
from models.vae import ConvVAE
from torch.utils.data import DataLoader
from lib.train_utils import create_optimizer, fetch_new_run, create_state
from torchvision.utils import save_image


def gaussian_distribution(y, mu, sigma):
    y = y.unsqueeze(1).expand_as(mu)
    result = MDN_CONST * torch.exp(-0.5 * ((y - mu) / sigma)**2) / sigma
    return result


def mdn_loss_function(out_pi, out_sigma, out_mu, y):
    result = gaussian_distribution(y, out_mu, out_sigma)
    result = result * out_pi
    result = torch.sum(result, dim=1)
    result = - torch.log(EPSILON + result)
    return torch.mean(result)


def train_epoch(lstm, optimizer, example):
    """ Used to train the 3 models over a single batch """

    optimizer.zero_grad()
    lstm.hidden = lstm.init_hidden()
    x = torch.cat((example['encoded'], example['actions'].view(-1, 1)), dim=1)
    x = x.view(-1, 1, x.size()[1])
    pi, sigma, mu = lstm(x)

    ex = example['encoded'][-1].view(1, example['encoded'].size()[1])
    target = torch.cat((example['encoded'][1:example['encoded'].size()[0]], ex,))
    loss = mdn_loss_function(pi, sigma, mu, target)
    loss.backward()
    optimizer.step()

    return float(loss)


def sample_long_term(vae, lstm, start_ex):
    frames = torch.tensor(start_ex[0], dtype=torch.float, device=DEVICE).div(255)
    save_image(frames.view(3, 128, 128), "results/test-origin.png")
    z = vae(frames.view(1, 3, 128, 128), encode=True)
    
    with torch.no_grad():
        for i in range(1, 100):
            new_state = torch.cat((z, torch.full((1, 1), 1, device=DEVICE)), dim=1)
            pi, sigma, mu = lstm(new_state.view(1, 1, 65))
            values, idx = torch.max(pi, 1)
            print(pi)
            z = mu[0][idx].view(1, 64)
            res = vae.decode(z)[0] 
            save_image(res, "results/test-{}.png".format(i))
    assert 0


def train_lstm(current_time):
    dataset = FrameDataset()
    last_id = 0
    lr = LR
    version = 1
    total_ite = 1
    # criterion = VAELoss()

    client = MongoClient()
    db = client.retro_contest
    collection = db[current_time]
    fs = gridfs.GridFS(db)

    vae, _ = load_model(current_time, -1, model="vae")
    if not vae:
        vae = ConvVAE((HEIGHT, WIDTH, 3), LATENT_VEC).to(DEVICE)

    lstm, checkpoint = load_model(current_time, -1, model="lstm")
    # lstm = False
    if not lstm:
        lstm = LSTM(HIDDEN_UNITS, LATENT_VEC,\
                     NUM_LAYERS, GAUSSIANS, HIDDEN_DIM).to(DEVICE)
        optimizer = create_optimizer(lstm, lr)
        state = create_state(version, lr, total_ite, optimizer)
        save_checkpoint(lstm, "lstm", state, current_time)
    else:
        optimizer = create_optimizer(vae, lr, param=checkpoint['optimizer'])
        total_ite = checkpoint['total_ite']
        lr = checkpoint['lr']
        version = checkpoint['version']
        last_id = 0
    # create_img(vae, version)
    
    while len(dataset) < SIZE:
        last_id = fetch_new_run(collection, fs, dataset, last_id, loaded_version=current_time)
        time.sleep(5)
    
    # sample_long_term(vae, lstm, dataset[1000])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    while True:
        batch_loss = []
        for batch_idx, (frames, actions, rewards) in enumerate(dataloader):
            running_loss = []
            # lr, optimizer = update_lr(lr, optimizer, total_ite)
            if total_ite % SAVE_TICK == 0:
                version += 1
                state = create_state(version, lr, total_ite, optimizer)
                save_checkpoint(lstm, "lstm", state, current_time)
                # create_img(lstm, version)
            frames = torch.tensor(frames, dtype=torch.float, device=DEVICE).div(255)
            encoded = vae(frames, encode=True)
            example = {
                'encoded': encoded,
                'actions' : torch.tensor(actions, dtype=torch.float, device=DEVICE)
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