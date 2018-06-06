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
        self.process_id = process_id
        self.game = game
        self.level = level
        self.current_time = current_time
        self.vae = vae
        self.lstm = lstm
        self.controller = controller
        self.result_queue = result_queue
        self.max_timestep = max_timestep
        self.convert = {0: 0, 1: 5, 2: 6, 3: 7}
    

    def _convert(self, predicted_actions):
        """ Convert predicted action into an environment action """

        predicted_actions = predicted_actions.numpy()[0]
        final_action = np.zeros((12,), dtype=np.bool)
        actions = np.where(predicted_actions > 0.5)[0]
        for i in actions:
            final_action[self.convert[i]] = True
        return final_action


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
            current_rewards = []

            while not done:
                with torch.no_grad():
                    if total_steps % SEQUENCE == 0:
                        self.lstm.hidden = self.lstm.init_hidden(1)

                    ## Predict the latent representation of the current frame
                    obs = torch.tensor(_formate_img(obs), dtype=torch.float, device=DEVICE).div(255)
                    z = self.vae(obs.view(1, 3, HEIGHT, WIDTH), encode=True)

                    ## Use the latent representation and the hidden state of the LSTM
                    ## to predict an action vector
                    actions = self.controller(torch.cat((z, 
                                self.lstm.hidden[0].view(1, -1),
                                self.lstm.hidden[1].view(1, -1)), dim=1))
                    final_action = self._convert(actions.cpu())
                    obs, reward, done, info = env.step(final_action)

                    ## Update the hidden state of the LSTM
                    action = torch.tensor(env.get_act(final_action), dtype=torch.float, device=DEVICE)\
                                            .div(ACTION_SPACE_DISCRETE)
                    lstm_input = torch.cat((z, action.view(1, 1)), dim=1) 
                    res = self.lstm(lstm_input.view(1, 1, LATENT_VEC + 1))

                ## Check for minimum reward duration the last buffer duration
                if len(current_rewards) == REWARD_BUFFER:
                    if np.mean(current_rewards) < MIN_REWARD:
                        break
                    current_rewards.insert(0, reward)
                    current_rewards.pop()
                else:
                    current_rewards.append(reward)
                total_reward += reward

                ## Check for rendering
                if (self.process_id + 1) % RENDER_TICK == 0:
                    if total_steps % 200 == 0:
                        print(actions)
                    env.render()
                
                ## Check for custom timelimit
                if total_steps > self.max_timestep:
                    break

                total_steps += 1
            final_reward.append(total_reward)

        final_time = timeit.default_timer() - start_time
        print("[{} / {}] Final mean reward: {}" \
                    .format(self.process_id + 1, POPULATION, np.mean(final_reward)))
        env.close()
        result = {}
        result[self.process_id] = (np.mean(final_reward), final_time)
        self.result_queue.put(result)
