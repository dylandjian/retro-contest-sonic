import gym
import numpy as np
from const import *

import gym_remote.client as grc
import gym_remote.exceptions as gre
from retro_contest.local import make
from utils import SonicDiscretizer, RewardScaler, WarpFrame, FrameStack


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



def create_env(test):
    env = make(ENV_NAME, STATE_NAME)
    # env = SonicDiscretizer(env)
    env = RewardScaler(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = TrackedEnv(env)
    return env

