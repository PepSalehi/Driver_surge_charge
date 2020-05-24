import os, sys
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '..'))
print(sys.path)
from env.MARL_env import MARLEnv

if __name__ == '__main__':
    env = MARLEnv()
    state = env._get_states()
    # print(state)
    for i in range(10):
        act_n = []
        obs_n, reward_n, done_n, _ = env.step(act_n)
