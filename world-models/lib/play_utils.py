from const import *
import timeit
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



def _formate_img(img):
    img = cv2.resize(np.array(img), dsize=(WIDTH, HEIGHT),\
                    interpolation=cv2.INTER_NEAREST)
    return img.transpose((2, 0, 1))


class JerkGame(multiprocessing.Process):
    def __init__(self, current_time, process_id):
        super(JerkGame, self).__init__()
        random.seed()
        self.id = process_id
        self.current_time = current_time
        self.levels = []
        self.frames = []
        self.actions = []
        self.rewards = []
        self.done = []



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
            exit(0)
            # self.levels = list(levels)
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
        ## Init possible levels
        game = GAMES["SONIC-1"]
        levels = LEVELS[game]

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
        self.levels = list(levels)
        current_level = self.get_level(levels)

        while True:

            ## Push in the database
            if current_idx > PLAYOUTS:
                self.add_db()
                total_frames += current_idx
                current_idx = 0

                if total_frames > PLAYOUTS_PER_LEVEL:
                    print("[PLAYING] Done with level: %s" % current_level)
                    total_frames = 0
                    current_level = self.get_level(levels)
                print("[PLAYING] Pushing the database for %d" % self.id)
                # time.sleep(180)
    
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
                    env = TrackedEnv(create_env(game, current_level))
                    _ = env.reset()
                    new_ep = False
            rew, new_ep = self.move(env, 100)
            current_idx += 100
            if not new_ep and rew <= 0:
                _, new_ep = self.move(env, 70, left=True)
            if new_ep:
                solutions.append(([max(env.reward_history)], env.best_sequence()))



class VAECGame(multiprocessing.Process):
    def __init__(self, current_time, process_id, vae, lstm, controller, game, level, result_queue, max_timestep):
        super(VAECGame, self).__init__()
        np.random.seed()
        self.process_id = process_id
        self.game = game
        self.level = level
        self.current_time = current_time
        self.vae = vae
        self.lstm = lstm
        self.controller = controller
        self.result_queue = result_queue
        self.max_timestep = max_timestep
    
    def run(self):
        final_reward = []
        env = False
        start_time = timeit.default_timer()
        for i in range(REPEAT_ROLLOUT):
            if env:
                env.close()
            env = create_env(self.game, self.level)
            obs = env.reset()
            done = False
            total_reward = 0
            total_steps = 0
            while not done:
                with torch.no_grad():
                    obs = torch.tensor(_formate_img(obs), dtype=torch.float, device=DEVICE).div(255)
                    z, _ = self.vae.encode(obs.view(1, 3, HEIGHT, WIDTH))
                    action = self.controller(torch.cat((z, 
                                self.lstm.hidden[0].view(1, -1),
                                self.lstm.hidden[1].view(1, -1)), dim=1))
                    obs, reward, done, info = env.step(int(action))
                    lstm_input = torch.cat((z, action), dim=1) 
                    _ = self.lstm(lstm_input.view(1, 1, LATENT_VEC + 1))
                total_steps += 1
                total_reward += reward
                if (self.process_id + 1) % RENDER_TICK == 0:
                    env.render()
                if total_steps > self.max_timestep:
                    break
            final_reward.append(total_reward)

        final_time = timeit.default_timer() - start_time
        if (self.process_id + 1) % RENDER_TICK == 0:
            print("[{} / {}] Final mean reward: {} <---- WHAT YOU ARE WATCHING"\
                        .format(self.process_id + 1, POPULATION, np.mean(final_reward)))
        else:
            print("[{} / {}] Final mean reward: {}" \
                    .format(self.process_id + 1, POPULATION, np.mean(final_reward)))
        env.close()
        result = {}
        result[self.process_id] = (np.mean(final_reward), final_time)
        self.result_queue.put(result)
