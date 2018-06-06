from retro_contest.local import make
import numpy as np


def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    total = 0
    while True:
        x = np.zeros((12,), dtype=np.bool)
        obs, rew, done, info = env.step(x)
        env.render()
        if done:
            obs = env.reset()
        total += 1


if __name__ == '__main__':
    main()