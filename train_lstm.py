import torch.nn.functional as F
import gridfs
import time
import torch
import numpy as np
import click
from const import *
from models.helper import load_model, save_checkpoint
from pymongo import MongoClient
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torchvision.utils import save_image
from models.helper import init_models
from lib.dataset import LSTMDataset
from lib.train_utils import create_optimizer, fetch_new_run, create_state
from lib.visu import sample_long_term


def mdn_loss_function(out_pi, out_sigma, out_mu, y):
    """
    Mixed Density Network loss function, see : 
    https://mikedusenberry.com/mixture-density-networks
    """

    result = Normal(loc=out_mu, scale=out_sigma)
    y = y.view(-1, SEQUENCE, 1, LATENT_VEC)
    result = torch.exp(result.log_prob(y))
    result = torch.sum(result * out_pi, dim=2)
    result = -torch.log(EPSILON + result)
    return torch.mean(result)


def train_epoch(lstm, optimizer, example):
    """ Used to train the 3 models over a single batch """

    ## Reset the gradients and the hidden state since we dont want to
    ## model sequence-wise relationships
    optimizer.zero_grad()
    lstm.hidden = lstm.init_hidden(SEQUENCE)

    ## Concatenate action to encoded vector
    x = torch.cat((example['encoded'],
            example['actions'].view(-1, 1) / ACTION_SPACE_DISCRETE), dim=1)
    x = x.view(-1, SEQUENCE, LATENT_VEC + 1)

    ## Shift target encoded vector
    last_ex = example['encoded'][-OFFSET].view(-1, example['encoded'].size()[1])
    target = torch.cat((example['encoded'][OFFSET:example['encoded'].size()[0]],\
                          last_ex,))

    pi, sigma, mu = lstm(x)
    loss = mdn_loss_function(pi, sigma, mu, target)
    loss.backward()
    optimizer.step()

    return float(loss)


def collate_fn(example):
    """ Custom way of flattening examples in a batch """

    frames = []
    actions = []

    for ex in example:
        frames.extend(ex[0])
        actions.extend(ex[1])

    frames = torch.tensor(frames, dtype=torch.float, device=DEVICE) / 255
    actions = torch.tensor(actions, dtype=torch.float, device=DEVICE) / (ACTION_SPACE_DISCRETE)
    return frames, actions


def train_lstm(current_time):
    """
    Train the LSTM to be able to predict the next latent vector z given the current vector z
    and an action.
    """

    dataset = LSTMDataset()
    client = MongoClient()
    db = client.retro_contest
    collection = db[current_time]
    fs = gridfs.GridFS(db)

    last_id = 0
    lr = LR
    version = 1
    total_ite = 1

    ## Load or create models
    vae, lstm, _, _, checkpoint = init_models(current_time, load_vae=True,
                load_lstm=True, load_controller=False)
    if not checkpoint:
        optimizer = create_optimizer(lstm, lr)
        state = create_state(version, lr, total_ite, optimizer)
        save_checkpoint(lstm, "lstm", state, current_time)
    else:
        optimizer = create_optimizer(lstm, lr, param=checkpoint['optimizer'])
        total_ite = checkpoint['total_ite']
        lr = checkpoint['lr']
        version = checkpoint['version']

    ## Fill the dataset (or wait for the database to be filled)
    while len(dataset) * PLAYOUTS < SIZE:
        last_id = fetch_new_run(collection, fs, dataset, last_id, loaded_version=current_time)
        time.sleep(5)
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_LSTM, collate_fn=collate_fn)
    while True:
        running_loss = []
        batch_loss = []

        for batch_idx, (frames, actions) in enumerate(dataloader):

            ## Save the model
            if total_ite % SAVE_TICK == 0:
                version += 1
                state = create_state(version, lr, total_ite, optimizer)
                save_checkpoint(lstm, "lstm", state, current_time)

            ## Save a picture of the long term sampling
            if total_ite % SAVE_PIC_TICK == 0:
                sample_long_term(vae, lstm, frames, version, total_ite)

            encoded = vae(frames, encode=True)
            example = {
                'encoded': encoded,
                'actions' : actions
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
