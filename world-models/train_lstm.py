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
    ## Prepare the input to be transformed into a gaussian distribution
    y = y.unsqueeze(1)
    y = y.expand(-1, GAUSSIANS, LATENT_VEC)

    result = MDN_CONST * torch.exp(-0.5 * ((y - mu) / sigma) ** 2) / sigma
    return result


def mdn_loss_function(out_pi, out_sigma, out_mu, y):
    result = gaussian_distribution(y, out_mu, out_sigma)
    result = result * out_pi
    result = torch.sum(result, dim=1)
    result =- torch.log(EPSILON + result)
    return torch.mean(result)


def train_epoch(lstm, optimizer, example):
    """ Used to train the 3 models over a single batch """

    optimizer.zero_grad()
    lstm.hidden = lstm.init_hidden()

    ## Concatenate action to encoded vector
    x = torch.cat((example['encoded'], example['actions'].view(-1, 1)), dim=1)
    x = x.view(-1, 1, x.size()[1])
    pi, sigma, mu = lstm(x)

    ## Shift all elements to the left and add a copy of the last
    ## element at the end
    last_ex = example['encoded'][-1].view(1, example['encoded'].size()[1])
    target = torch.cat((example['encoded'][1:example['encoded'].size()[0]],\
                         last_ex,))

    loss = mdn_loss_function(pi, sigma, mu, target)
    loss.backward()
    optimizer.step()

    return float(loss)


def sample(pi, mu, sigma):
    pi = pi.cpu().numpy()[0]
    mu = mu.cpu().numpy()[0]
    sigma = sigma.cpu().numpy()[0]

    ## Get the correct index depending on the probabilities
    ## of each Gaussian distribution 
    z = np.random.gumbel(loc=0, scale=1, size=pi.shape)
    k = (np.log(pi) + z)
    k = k.argmax(axis=1)

    ## Create a Gaussian distribution depending on the k
    ## above, scaling by sigma and offsetting mu
    rn = np.random.randn(LATENT_VEC)
    sampled = rn * sigma[k][0] + mu[k][0]
    sampled_tensor = torch.tensor(sampled, dtype=torch.float, device=DEVICE)
    return sampled_tensor.view(1, LATENT_VEC)


def sample_long_term(vae, lstm, start_ex):
    """ Given a frame, tries to predict the next 100 encoded vectors """

    frames = torch.tensor(start_ex[0], dtype=torch.float, device=DEVICE).div(255)
    save_image(frames.view(3, WIDTH, HEIGHT), "results/test-origin.png")
    old_z = vae(frames.view(1, 3, WIDTH, HEIGHT), encode=True)
    
    with torch.no_grad():
        for i in range(1, 100):
            if i == 1:
                new_state = torch.cat((old_z, torch.full((1, 1), 1, device=DEVICE)), dim=1)
            else:
                new_state = torch.cat((z, torch.full((1, 1), 1, device=DEVICE)), dim=1)
            pi, sigma, mu = lstm(new_state.view(1, 1, LATENT_VEC + 1))
            z = sample(pi, mu, sigma)
            res = vae.decode(z)[0] 
            print(res)
            save_image(res, "results/test-{}.png".format(i))
    assert 0


def train_lstm(current_time):
    dataset = FrameDataset()
    last_id = 0
    lr = LR
    version = 1
    total_ite = 1

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
        optimizer = create_optimizer(lstm, lr, param=checkpoint['optimizer'])
        total_ite = checkpoint['total_ite']
        lr = checkpoint['lr']
        version = checkpoint['version']
        last_id = 0
    
    while len(dataset) < SIZE:
        last_id = fetch_new_run(collection, fs, dataset, last_id, loaded_version=current_time)
        time.sleep(5)
    
    sample_long_term(vae, lstm, dataset[1000])
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    while True:
        batch_loss = []
        for batch_idx, (frames, actions, rewards) in enumerate(dataloader):
            running_loss = []
            # lr, optimizer = update_lr(lr, optimizer, total_ite)

            ## Save the model
            if total_ite % SAVE_TICK == 0:
                version += 1
                state = create_state(version, lr, total_ite, optimizer)
                save_checkpoint(lstm, "lstm", state, current_time)

            ## Create input tensors
            frames = torch.tensor(frames, dtype=torch.float, device=DEVICE).div(255)
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