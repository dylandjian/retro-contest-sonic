import numpy as np
from torch.utils.data import Dataset
from const import *
import matplotlib.pyplot as plt


class VAEDataset(Dataset):

    def __init__(self, size=SIZE, repeat=REPEAT):
        """ Instanciate a dataset extending PyTorch """

        self.repeat = repeat
        self.current_len = 0
        self.frames = np.zeros((size, 3, HEIGHT, WIDTH))


    def __len__(self):
        return self.current_len


    def __getitem__(self, idx):
        return self.frames[idx]


    def update(self, run):
        """ Rotate the circular buffer to add new games at end """

        total_obs = len(run[0])
        self.current_len = min(self.current_len + total_obs, SIZE)
        self.frames = np.roll(self.frames, total_obs, axis=0)
        self.frames[:total_obs] = np.array(run[0])

        return total_obs 


class LSTMDataset(Dataset):

    def __init__(self, lstm=False, size=SIZE, repeat=REPEAT):
        """ Instanciate a dataset extending PyTorch """

        self.repeat = repeat
        self.frames = []
        self.actions = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        sample_idx = np.random.randint(0, PLAYOUTS - SAMPLE_SIZE)
        return self.frames[idx][sample_idx:sample_idx + SAMPLE_SIZE], \
                    self.actions[idx][sample_idx:sample_idx + SAMPLE_SIZE]


    def update(self, run):
        """ Rotate the circular buffer to add new games at end """

        total_obs = len(run[0])

        if len(self.frames) * PLAYOUTS > SIZE:
            self.frames.pop(0)
        self.frames.append(np.array(run[0]))

        if len(self.actions) * PLAYOUTS > SIZE:
            self.actions.pop(0)
        self.actions.append(np.array(run[1]))

        # self.rewards = np.roll(self.rewards, total_obs, axis=0)
        # self.rewards[:total_obs] = np.array(run[2])

        return total_obs 
