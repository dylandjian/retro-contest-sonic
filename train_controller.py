import click
import pickle
import os
import numpy as np
import time
import torch.multiprocessing as multiprocessing
from const import *
from torch.multiprocessing import Queue
from models.helper import save_checkpoint, init_models
from lib.agent_play import VAECGame
from models.controller import Controller


def compute_ranks(x):
  """
  Returns ranks in [0, len(x))
  Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
  (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks


def rankmin(x):
  """
  https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= .5
  return y


def init_controller(controller, solution):
    """ Change the weights of the controller by the one proposed by the CMA """

    new_w1 = torch.tensor(solution[0:PARAMS_FC1 + LATENT_VEC],\
                                dtype=torch.double, device=DEVICE)
    new_w2 = torch.tensor(solution[PARAMS_FC1 + LATENT_VEC:],\
                                dtype=torch.double, device=DEVICE)
    params = controller.state_dict() 
    params['fc1.weight'].data.copy_(new_w1)
    params['fc2.weight'].data.copy_(new_w2)
    return


def create_results(result_queue, fitlist):
    """ Empty the result queue to adapt the fitlst, and get the timers for display """

    times = []
    for i in range(POPULATION):
        result = result_queue.get()
        keys = list(result.keys())
        result = list(result.values())
        fitlist[keys[0]] = result[0][0]
        times.append(result[0][1])
    return times


def train_controller(current_time):
    """
    Train the controllers by using the CMA-ES algorithm to improve candidature solutions
    by testing them in parallel using multiprocessing
    """

    current_time = str(current_time)
    number_generations = 1
    games = GAMES
    levels = LEVELS
    current_game = False
    result_queue = Queue()

    vae, lstm, best_controller, solver, checkpoint = init_models(current_time, sequence=1,
                                        load_vae=True, load_controller=True, load_lstm=True)
    if checkpoint:
        current_ctrl_version = checkpoint["version"]
        current_solver_version = checkpoint["solver_version"]
        new_results = solver.result()
        current_best = new_results[1]
    else:
        current_ctrl_version = 1
        current_solver_version = 1
        current_best = 0

    while True:
        solutions = solver.ask()
        fitlist = np.zeros(POPULATION)
        eval_left = 0
    
        ## Once a level is beaten, remove it from the training set of levels
        if current_best > SCORE_CAP or not current_game:
            if not current_game or len(levels[current_game]) == 0:
                current_game = games[0]
                games.remove(current_game)
                current_best = 0
            current_level = np.random.choice(levels[current_game])
            levels[current_game].remove(current_level)

        print("[CONTROLLER] Current game: %s and level is: %s" % (current_game, current_level))
        while eval_left < POPULATION:
            jobs = []
            todo = PARALLEL if eval_left + PARALLEL <= POPULATION else (eval_left + PARALLEL) % POPULATION

            ## Create the child processes to evaluate in parallel
            print("[CONTROLLER] Starting new batch")
            for job in range(todo):
                process_id = eval_left + job

                ## Assign new weights to the controller, given by the CMA
                controller = Controller(LATENT_VEC, PARAMS_FC1, ACTION_SPACE).to(DEVICE)
                init_controller(controller, solutions[process_id])

                ## Start the evaluation
                new_game = VAECGame(process_id, vae, lstm, controller,
                                    current_game, current_level, result_queue)
                new_game.start()
                jobs.append(new_game)

            ## Wait for the evaluation to be completed
            for p in jobs:
                p.join()

            eval_left = eval_left + todo
            print("[CONTROLLER] Done with batch")

        ## Get the results back from the processes 
        times = create_results(result_queue, fitlist)

        ## For display
        current_score = np.max(fitlist)
        average_score = np.mean(fitlist)

        ## Update solver with results
        max_idx = np.argmax(fitlist)
        fitlist = rankmin(fitlist)
        solver.tell(fitlist)
        new_results = solver.result()

        ## Display
        print("[CONTROLLER] Total duration for generation: %.3f seconds, average duration:"
            " %.3f seconds per process, %.3f seconds per run" % ((np.sum(times), \
                    np.mean(times), np.mean(times) / REPEAT_ROLLOUT)))
        print("[CONTROLLER] Creating generation: {} ...".format(number_generations + 1))
        print("[CONTROLLER] Current best score: {}, new run best score: {}".format(current_best, current_score))
        print("[CONTROLLER] Best score ever: {}, current number of improvements: {}".format(current_best, current_ctrl_version))
        print("[CONTROLLER] Average score on all of the processes: {}\n".format(average_score))
    
        ## Save the new best controller
        if current_score > current_best:
            init_controller(best_controller, solutions[max_idx])
            state = { 'version': current_ctrl_version,
                      'solver_version': current_solver_version,
                      'score': current_score,
                      'level': current_level,
                      'game': current_game,
                      'generation': number_generations }
            save_checkpoint(best_controller, "controller", state, current_time)
            current_ctrl_version += 1
            current_best = current_score

        ## Save solver and change level to a random one
        if number_generations % SAVE_SOLVER_TICK == 0:
            dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), \
                        'saved_models', current_time, "{}-solver.pkl".format(current_solver_version))
            pickle.dump(solver, open(dir_path, 'wb'))
            current_solver_version += 1
            current_level = np.random.choice(levels[current_game])

        number_generations += 1

        
@click.command()
@click.option("--folder", default=-1)
def main(folder):
    multiprocessing.set_start_method('spawn')
    if folder == -1:
        current_time = int(time.time())
    else:
        current_time = folder
    train_controller(current_time)
        


if __name__ == "__main__":
    main()
