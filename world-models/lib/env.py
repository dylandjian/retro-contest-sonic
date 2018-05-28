from const import *

import gym
import gym_remote.client as grc
import numpy as np
from retro_contest.local import make



def create_env(env_name, env_state, contest=False):
    if not contest:
        env = SonicDiscretizer(make(env_name, env_state))
    else:
        env = grc.RemoteEnv('tmp/sock')
    return env


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
        self.action_space = gym.spaces.Discrete(len(self._actions))


    def action(self, a):
        if isinstance(a, np.ndarray):
            return a.copy()
        return self._actions[a].copy()
    
    
    def get_act(self, a):
        for i in range(len(self._actions) - 1):
            if np.array_equal(self._actions[i], a):
                return i
