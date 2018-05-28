import click
from const import *
from lib.play_utils import JerkGame
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

    jobs = [JerkGame(str(current_time), process_id) for process_id\
                    in range(PARALLEL)]
    for p in jobs:
        p.start()
    for p in jobs:
        p.join()
        


if __name__ == "__main__":
    main()