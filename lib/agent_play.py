import numpy as np
import torch.multiprocessing as multiprocessing
import timeit
import time
import torch
from .env import create_env
from .play_utils import _formate_img
from const import *


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
        convert = {0: 0, 1: 5, 2: 6, 3: 7}
        for i in range(REPEAT_ROLLOUT):
            if env:
                env.close()
            env = create_env(self.game, self.level)
            obs = env.reset()
            done = False
            total_reward = 0
            total_steps = 0
            current_rewards = []
            while not done:
                with torch.no_grad():
                    obs = torch.tensor(_formate_img(obs), dtype=torch.float, device=DEVICE).div(255)
                    z, _ = self.vae.encode(obs.view(1, 3, HEIGHT, WIDTH))
                    # action = self.controller(z)
                    action = self.controller(torch.cat((z, 
                                self.lstm.hidden[0].view(1, -1),
                                self.lstm.hidden[1].view(1, -1)), dim=1))
                    actions = action.cpu().numpy()[0]
                    final_action = np.zeros((12,), dtype=np.bool)
                    res = np.where(actions > 0.5)[0]
                    for i in res:
                        final_action[convert[i]] = True
                    obs, reward, done, info = env.step(final_action)
                    action = torch.tensor(env.get_act(final_action), dtype=torch.float, device=DEVICE).div(10)
                    lstm_input = torch.cat((z, action.view(1, 1)), dim=1) 
                    future = self.lstm(lstm_input.view(1, 1, LATENT_VEC + 1))
                total_steps += 1
                if len(current_rewards) == REWARD_BUFFER:
                    if np.mean(current_rewards) < MIN_REWARD:
                        break
                    current_rewards.insert(0, reward)
                    current_rewards.pop()
                else:
                    current_rewards.append(reward)
                total_reward += reward
                if (self.process_id + 1) % RENDER_TICK == 0:
                    env.render()
                if total_steps > self.max_timestep:
                    break
            final_reward.append(total_reward)

        final_time = timeit.default_timer() - start_time
        if (self.process_id + 1) % RENDER_TICK == 0:
            print("[{} / {}] Final mean reward: {} <---- WHAT YOU WERE WATCHING"\
                        .format(self.process_id + 1, POPULATION, np.mean(final_reward)))
        else:
            print("[{} / {}] Final mean reward: {}" \
                    .format(self.process_id + 1, POPULATION, np.mean(final_reward)))
        env.close()
        result = {}
        result[self.process_id] = (np.mean(final_reward), final_time)
        self.result_queue.put(result)
