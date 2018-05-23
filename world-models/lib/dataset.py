import numpy as np
from torch.utils.data import Dataset
from const import *
import matplotlib.pyplot as plt


class FrameDataset(Dataset):

    def __init__(self, size=SIZE, repeat=REPEAT):
        """ Instanciate a dataset """

        self.frames = np.zeros((size, 3, HEIGHT, WIDTH))
        self.actions = np.zeros((size))
        self.rewards = np.zeros((size))
        self.current_len = 0
        self.repeat = repeat


    def __len__(self):
        return self.current_len


    def __getitem__(self, idx):
        
        return self.frames[idx], \
            self.actions[idx], self.rewards[idx]


    def _formate_state(self, frames, actions, rewards):
        """ Repeat the probas and the winner to make every example identical after
            the dihedral rotation has been applied """

        actions = np.full((self.repeat, 1), actions)
        rewards = np.full((self.repeat, 1), rewards)
        return frames, actions, rewards


    def update(self, run):
        """ Rotate the circular buffer to add new games at end """

        total_obs = len(run[0])
        
        self.current_len = min(self.current_len + total_obs, SIZE)

        self.frames = np.roll(self.frames, total_obs, axis=0)
        self.frames[:total_obs] = np.array(run[0])

        self.actions = np.roll(self.actions, total_obs, axis=0)
        self.actions[:total_obs] = np.array(run[1])

        self.rewards = np.roll(self.rewards, total_obs, axis=0)
        self.rewards[:total_obs] = np.array(run[2])

        return total_obs 
