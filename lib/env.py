from const import *

import gym
import gym_remote.client as grc
import numpy as np
from retro_contest.local import make
import retro


def create_env(env_name, env_state, contest=False, human=False):
    if human:
        env = SonicDiscretizer(retro.make(env_name, env_state, scenario="contest", use_restricted_actions=retro.ACTIONS_FILTERED))
    elif not contest:
        env = SonicDiscretizer(make(env_name, env_state))
    else:
        env = SonicDiscretizer(grc.RemoteEnv('tmp/sock'))
    return env


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


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['LEFT', 'B'], ['RIGHT', 'B'], ['B']]
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self._actions.append(np.zeros((12,), dtype=np.bool))
        self.action_space = gym.spaces.Discrete(len(self._actions))


    def filter_act(self, a):
        """ Removing weird combos of buttons / useless buttons """

        a[4] = False
        a[9] = False
        a[10] = False
        a[11] = False
        if a[6] == True and a[7] == True:
            a[6] = False
            a[7] = False
        if a[1] == True:
            a[0] = True
            a[1] = False
        if a[8] == True:
            a[0] = True
            a[8] = False
        if a[0] == True and (a[6] == True or a[7] == True):
            a[5] = False
        return a

    def action(self, a):
        if isinstance(a, np.ndarray):
            a = self.filter_act(a)
            return a.copy()
        return self._actions[a].copy()
    
    
    def get_act(self, a):
        a = self.filter_act(a)
        for i in range(len(self._actions)):
            if np.array_equal(self._actions[i], a):
                return i
