from const import *
from .env import create_env


def play(game, level):
    frames = []
    actions = []
    rewards = []

    env = create_env(game, level)
    obs = env.reset()
    done = False

    for _ in range(PLAYOUTS // PARALLEL):
        if done:
            obs = env.reset()
        action = env.action_space.sample()
        print("action %d" % action)
        frames.append(obs)
        obs, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)

    return frames, actions, rewards