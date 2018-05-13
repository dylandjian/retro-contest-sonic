import numpy as np
from torch.utils.data import Dataset
from const import *




class FrameDataset(Dataset):
    """
    """

    def __init__(self, size=SIZE, repeat=REPEAT):
        """ Instanciate a dataset """

        self.frames = np.zeros((size, repeat, HEIGHT, WIDTH, 3))
        self.actions = np.zeros((size))
        self.reward = np.zeros((size))
        self.current_len = 0
        self.repeat = repeat


    def __len__(self):
        return self.current_len


    def __getitem__(self, idx):
        return self._formate_state(self.frames[idx], \
            self.action[idx], self.reward[idx])


    def _formate_state(frames, actions, reward):
        """ Repeat the probas and the winner to make every example identical after
            the dihedral rotation has been applied """

        actions = np.full((self.repeat, 1), actions)
        reward = np.full((self.repeat, 1), actions)
        return state, actions, reward


    def update(self, run):
        """ Rotate the circular buffer to add new games at end """

        return number_moves
