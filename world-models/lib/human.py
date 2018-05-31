from .play_utils import _formate_img
from .env import create_env
import pickle
import multiprocessing
import random
import gridfs
import numpy as np
from pymongo import MongoClient
import os
from const import *


class HumanGame(multiprocessing.Process):
    def __init__(self, current_time, process_id, game):
        super(HumanGame, self).__init__()
        random.seed()
        self.id = process_id
        self.current_time = current_time
        self.frames = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.game = game
        self.levels = LEVELS


    def add_db(self):
        file_id = self.fs.put(pickle.dumps([self.frames, self.actions,\
                     self.rewards, self.done]))
        obs_id = self.collection.find().count()
        self.collection.insert({
            "id": obs_id,
            "run": file_id
        })
        self.frames.clear()
        self.actions.clear()
        self.rewards.clear()
        return


    def get_level(self):
        if len(self.levels[self.game]) == 0:
            exit(0)
        level = random.choice(self.levels[self.game])
        self.levels[self.game].remove(level)
        return level
    

    def run(self):
        ## Init possible levels

        current_idx = 0
        total_frames = 0
        env = False

        ## Database connection
        client = MongoClient()
        db = client.retro_contest

        ## Put the database variable inside the class instance
        self.collection = db[self.current_time]
        self.fs = gridfs.GridFS(db)
        current_level = self.get_level()

        while True:

            movie_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    '..', 'replay/{}/contest/{}-{}-0000.bk2'.format(self.game, self.game, current_level))
            movie = retro.Movie(movie_path)
            movie.step()

            if env:
                env.close()
            env = create_env(self.game, current_level)
            env.initial_state = movie.get_state()
            env.reset()

            steps_taken = 0
            done = False

            movie.step()
            while True:

                ## Move
                action = np.zeros((12,), dtype=np.bool)
                for i in range(12):
                    action[i] = movie.get_key(i)
                obs, rew, done, info = env.step(action)

                ## Add move to observations
                if steps_taken > 0 and steps_taken <= PLAYOUTS:
                    self.actions.append(env.get_act(action))
                    self.rewards.append(rew)
                    self.done.append(done)
                
                if done or not movie.step():
                    break

                steps_taken += 1                    
                if steps_taken <= PLAYOUTS:
                    self.frames.append(_formate_img(obs))
                if steps_taken > PLAYOUTS or done:
                    self.add_db()
                    total_frames += steps_taken
                    steps_taken = 0

                    if total_frames > PLAYOUTS_PER_LEVEL:
                        print("[PLAYING] Done with level: %s, %d level left in game %s" \
                                % (current_level, len(self.levels[self.game]), self.game))
                        total_frames = 0
                        current_level = self.get_level()
                        break
    
