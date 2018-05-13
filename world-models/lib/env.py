from const import *

import gym_remote.client as grc
from retro_contest.local import make



def create_env(env_name, env_state, contest=False):
    if not contest:
        env = make(env_name, env_state, discrete_actions=True)
    else:
        env = grc.RemoteEnv('tmp/sock')
    return env
