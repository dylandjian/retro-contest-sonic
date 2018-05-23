from retro_contest.local import make


def main():
    env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
    obs = env.reset()
    while True:
        obs, rew, done, info = env.step(1)
        env.render()
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()