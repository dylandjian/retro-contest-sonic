import numpy as np
import torch.multiprocessing as multiprocessing
import timeit
import time
import torch
from .env import create_env
from .controller_utils import _formate_img
from const import *


class VAECGame(multiprocessing.Process):
    def __init__(self, process_id, vae, lstm, controller, game, level, result_queue):
        super(VAECGame, self).__init__()
        self.process_id = process_id
        self.game = game
        self.level = level
        self.vae = vae
        self.lstm = lstm
        self.controller = controller
        self.result_queue = result_queue
        self.convert = {0: 0, 1: 5, 2: 6, 3: 7}
    

    def _convert(self, predicted_actions):
        """ Convert predicted action into an environment action """

        ## Transform the sigmoid output vector
        final_action = np.zeros((12,), dtype=np.bool)
        not_actions = torch.full((predicted_actions.size(0),), -1, device=DEVICE)
        actions = torch.where(predicted_actions >= 0.5, predicted_actions, not_actions)[0]

        ## Convert indexes to buttons on the SEGA controller
        for idx in range(actions.size(0)):
            if actions[idx] != -1.:
                final_action[self.convert[idx]] = True

        return final_action


    def run(self):
        """ Called by process.start() """

        final_reward = []
        start_time = timeit.default_timer()
        env = create_env(self.game, self.level)

        for _ in range(REPEAT_ROLLOUT):
            obs = env.reset()
            done = False
            total_reward = 0
            total_steps = 0
            current_rewards = []

            while not done:
                with torch.no_grad():

                    ## Reset the hidden state once we have seen SEQUENCE number of frames
                    if total_steps % SEQUENCE == 0:
                        self.lstm.hidden = self.lstm.init_hidden(1)

                    ## Predict the latent representation of the current frame
                    obs = torch.tensor(_formate_img(obs), dtype=torch.float, device=DEVICE).div(255)
                    z = self.vae(obs.view(1, 3, HEIGHT, WIDTH), encode=True)

                    ## Use the latent representation and the hidden state and cell of
                    ## the LSTM to predict an action vector
                    actions = self.controller(torch.cat((z, 
                                self.lstm.hidden[0].view(1, -1),
                                self.lstm.hidden[1].view(1, -1)), dim=1))
                    final_action = self._convert(actions)
                    obs, reward, done, info = env.step(final_action)

                    ## Update the hidden state and cell of the LSTM
                    action = torch.tensor(env.get_act(final_action), dtype=torch.float, device=DEVICE)\
                                            .div(ACTION_SPACE_DISCRETE)
                    lstm_input = torch.cat((z, action.view(1, 1)), dim=1) 
                    _ = self.lstm(lstm_input.view(1, 1, LATENT_VEC + 1))

                ## Check for minimum reward duration the last buffer duration
                if len(current_rewards) == REWARD_BUFFER:
                    if np.mean(current_rewards) < MIN_REWARD:
                        break
                    current_rewards.insert(0, reward)
                    current_rewards.pop()
                else:
                    current_rewards.append(reward)

                ## Check for rendering for debug / fun
                if (self.process_id + 1) % RENDER_TICK == 0:
                    env.render()

                total_reward += reward
                total_steps += 1

            final_reward.append(total_reward)

        final_time = timeit.default_timer() - start_time
        print("[{} / {}] Final mean reward: {}" \
                    .format(self.process_id + 1, POPULATION, np.mean(final_reward)))
        env.close()
        result = {}
        result[self.process_id] = (np.mean(final_reward), final_time)
        self.result_queue.put(result)
