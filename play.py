import click
from const import *
from lib.human import HumanGame
import multiprocessing
import time


@click.command()
@click.option("--folder", default=-1)
@click.option("--contest/--no_contest", default=False)
def main(contest, folder):
    multiprocessing.set_start_method('spawn')
    if folder == -1:
        current_time = int(time.time())
    else:
        current_time = folder

    jobs = []
    i = 1
    for game in GAMES:
        for j in range(1, PARALLEL_PER_GAME + 1):
            jobs.append(HumanGame(str(current_time), j * i, game))
        i += 1
    for p in jobs:
        p.start()
    for p in jobs:
        p.join()
        


if __name__ == "__main__":
    main()