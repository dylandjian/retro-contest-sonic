import numpy as np
from torch.utils.data import Dataset
from const import *
import matplotlib.pyplot as plt


class FrameDataset(Dataset):

    def __init__(self, lstm=False, size=SIZE, repeat=REPEAT):
        """ Instanciate a dataset extending PyTorch """

        self.lstm = lstm
        self.repeat = repeat

        self.frames = []
        if lstm:
            self.actions = []

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):

        sample_idx = np.random.randint(0, PLAYOUTS - SAMPLE_SIZE)
        if self.lstm:
            return self.frames[idx][sample_idx:sample_idx + SAMPLE_SIZE], \
                            self.actions[idx][sample_idx:sample_idx + SAMPLE_SIZE]
        return self.frames[idx][sample_idx:sample_idx + SAMPLE_SIZE]


    def update(self, run):
        """ Rotate the circular buffer to add new games at end """

        total_obs = len(run[0])

        if len(self.frames) * PLAYOUTS > SIZE:
            self.frames.pop(0)
        self.frames.append(np.array(run[0]))

        if self.lstm:
            if len(self.actions) * PLAYOUTS > SIZE:
                self.actions.pop(0)
            self.actions.append(np.array(run[1]))

        # self.rewards = np.roll(self.rewards, total_obs, axis=0)
        # self.rewards[:total_obs] = np.array(run[2])

        return total_obs 
