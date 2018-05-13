import click
from lib.env import create_env
from const import *
import numpy as np
from models.vae import VAE
from torch import nn, optim
from lib.dataset import FrameDataset
from lib.train import train
from lib.play import play
import threading
import multiprocessing


@click.command()
@click.option("--contest/--no_contest", default=False)
def main(contest):
    multiprocessing.set_start_method('spawn')
    dataset = FrameDataset()
    vae = VAE((HEIGHT, WIDTH, 3), 100, 40).to(DEVICE)
    train_thread = threading.Thread(target=train, args=(dataset, vae,))

    game = GAMES["SONIC-1"]
    level = LEVELS[game][5]
    pool = multiprocessing.Pool(processes=PARALLEL)
    while True:
        res = [pool.apply(play, args=(game,LEVELS[game][i * 2],)) for i in range(PARALLEL)]
        print(len(res))
        assert 0

    train_thread.join()



if __name__ == "__main__":
    main()