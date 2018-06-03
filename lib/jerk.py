import numpy as np
import random
import multiprocessing
import gridfs
import pickle
from .env import create_env
from .play_utils import _formate_img
from pymongo import MongoClient
from const import *



class JerkGame(multiprocessing.Process):
    def __init__(self, current_time, process_id, game):
        super(JerkGame, self).__init__()
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
                self.frames.append(_formate_img(obs))
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
        current_idx = 0
        total_frames = 0
        env = False
        new_ep = True
        solutions = []

        ## Database connection
        client = MongoClient()
        db = client.retro_contest

        ## Put the database variable inside the class instance
        self.collection = db[self.current_time]
        self.fs = gridfs.GridFS(db)
        current_level = self.get_level()

        while True:

            ## Push in the database
            if current_idx > PLAYOUTS:
                self.add_db()
                total_frames += current_idx
                current_idx = 0

                if total_frames > PLAYOUTS_PER_LEVEL:
                    print("[PLAYING] Done with level: %s, %d level left in game %s" \
                            % (current_level, len(self.levels[self.game]), self.game))
                    total_frames = 0
                    current_level = self.get_level()
                print("[PLAYING] Pushing the database for %d" % self.id)
    
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
                    env = TrackedEnv(create_env(self.game, current_level))
                    _ = env.reset()
                    new_ep = False
            rew, new_ep = self.move(env, 100)
            current_idx += 100
            if not new_ep and rew <= 0:
                _, new_ep = self.move(env, 70, left=True)
            if new_ep:
                solutions.append(([max(env.reward_history)], env.best_sequence()))
