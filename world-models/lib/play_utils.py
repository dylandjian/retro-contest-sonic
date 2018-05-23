from const import *
import matplotlib.pyplot as plt
import numpy as np
from .env import create_env
import multiprocessing
import pickle
import random
import cv2
from pymongo import MongoClient
import gym
import numpy as np
import time
import gridfs

class TrackedEnv(gym.Wrapper):
    """
    An environment that tracks the current trajectory and
    the total number of timesteps ever taken.
    """
    def __init__(self, env):
        super(TrackedEnv, self).__init__(env)
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        self.total_steps_ever = 0

    def best_sequence(self):
        """
        Get the prefix of the trajectory with the best
        cumulative reward.
        """
        max_cumulative = max(self.reward_history)
        for i, rew in enumerate(self.reward_history):
            if rew == max_cumulative:
                return self.action_history[:i+1]
        raise RuntimeError('unreachable')

    def reset(self, **kwargs):
        self.action_history = []
        self.reward_history = []
        self.total_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.total_steps_ever += 1
        self.action_history.append(action.copy())
        obs, rew, done, info = self.env.step(action)
        self.total_reward += rew
        self.reward_history.append(self.total_reward)
        return obs, rew, done, info
    
    def get_act(self, a):
        return self.env.get_act(a)


class Game(multiprocessing.Process):
    def __init__(self, current_time, process_id, mode="jerk"):
        super(Game, self).__init__()
        random.seed()
        self.id = process_id
        self.current_time = current_time
        self.levels = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.done = []
        self.mode = mode


    def _formate_img(self, img):
        img = cv2.resize(np.array(img), dsize=(WIDTH, HEIGHT),\
                     interpolation=cv2.INTER_NEAREST)
        return img.transpose((2, 0, 1))


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


    def get_level(self, levels):
        if len(self.levels) == 0:
            self.levels = list(levels)
        level = random.choice(self.levels)
        self.levels.remove(level)
        return level
    

    def move(self, env, num_steps, left=False, jump_prob=1.0 / 10.0, jump_repeat=4):
        """
        Move right or left for a certain number of steps,
        jumping periodically.
        """
        total_rew = 0.0
        done = False
        steps_taken = 0
        jumping_steps_left = 0
        while not done and steps_taken < num_steps:
            action = np.zeros((12,), dtype=np.bool)
            action[6] = left
            action[7] = not left
            if jumping_steps_left > 0:
                action[0] = True
                jumping_steps_left -= 1
            else:
                if random.random() < jump_prob:
                    jumping_steps_left = jump_repeat - 1
                    action[0] = True
            obs, rew, done, _ = env.step(action)
            if steps_taken > 0:
                self.actions.append(env.get_act(action))
                self.rewards.append(rew)
                self.done.append(done)
            total_rew += rew
            steps_taken += 1
            if done:
                break
            if steps_taken < num_steps:
                self.frames.append(self._formate_img(obs))
        return total_rew, done


    def exploit(self, env, sequence):
        """
        Replay an action sequence; pad with NOPs if needed.
        Returns the final cumulative reward.
        """
        env.reset()
        done = False
        idx = 0
        while not done:
            if idx >= len(sequence):
                _, _, done, _ = env.step(np.zeros((12,), dtype='bool'))
            else:
                _, _, done, _ = env.step(sequence[idx])
            idx += 1
        return env.total_reward


    def run(self):
        ## Init possible levels
        game = GAMES["SONIC-1"]
        levels = LEVELS[game]

        current_idx = 0
        env = False
        new_ep = True
        solutions = []

        ## Database connection
        client = MongoClient()
        db = client.retro_contest

        ## Put the database variable inside the class instance
        self.collection = db[self.current_time]
        self.fs = gridfs.GridFS(db)
        self.levels = list(levels)


        while True:

            ## Push in the database
            if current_idx > PLAYOUTS:
                self.add_db()
                current_idx = 0
                print("[PLAYING] Pushing the database for %d" % self.id)
                # time.sleep(180)
    
            if self.mode == "jerk":
                if new_ep:
                    if (solutions and
                            random.random() < EXPLOIT_BIAS + env.total_steps_ever / TOTAL_TIMESTEPS):
                        solutions = sorted(solutions, key=lambda x: np.mean(x[0]))
                        best_pair = solutions[-1]
                        new_rew = self.exploit(env, best_pair[1])
                        best_pair[0].append(new_rew)
                        continue
                    else:
                        if env:
                            env.close()
                        env = TrackedEnv(create_env(game, self.get_level(levels)))
                        _ = env.reset()
                        new_ep = False
                rew, new_ep = self.move(env, 100)
                current_idx += 100
                if not new_ep and rew <= 0:
                    _, new_ep = self.move(env, 70, left=True)
                if new_ep:
                    solutions.append(([max(env.reward_history)], env.best_sequence()))
            elif mode == "vaec":
                pass
