import click
import pickle
import os
import numpy as np
from const import *
import torch.multiprocessing as multiprocessing
from torch.multiprocessing import Queue
import time
from models.helper import save_checkpoint
from lib.controller_utils import CMAES
from lib.agent_play import VAECGame
from lib.train_utils import init_models



def test_controller(current_time):
    current_time = str(current_time)
    games = GAMES
    levels = LEVELS
    result_queue = Queue()

    vae, lstm, best_controller, solver, checkpoint = init_models(current_time)
    print("Score: %d" % checkpoint['score'])
    game = games[0]
    level = levels[game][0]
    print("[CONTROLLER] Current level is: %s" % level)
    new_game = VAECGame(current_time, 0, vae, lstm, best_controller, \
            game, level, result_queue, 4500)
    new_game.start()
    new_game.join()

        
@click.command()
@click.option("--folder", default=-1)
def main(folder):
    multiprocessing.set_start_method('spawn')
    if folder == -1:
        current_time = int(time.time())
    else:
        current_time = folder
    test_controller(current_time)


if __name__ == "__main__":
    main()
