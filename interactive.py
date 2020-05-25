import os, sys
import numpy as np
import matplotlib.pyplot as plt

from env.MARL_env import MultiAgentEnv

if __name__ == '__main__':
    env = MultiAgentEnv()
    # state = env._get_states()
    # print(state)
    rewards = []
    print("time at the start ", env.T)
    done_n = False
    while not done_n:
        act_n = {z.id: np.random.random() for z in env.model.zones}
        obs_n, reward_n, done_n, _ = env.step(act_n)
        rewards.append(reward_n)
        if reward_n > 0:
            print("reward was greater than zero at t ", env.T)
        print("reward is ", reward_n)
    print ("time at the end ", env.T)
    print(rewards)
    plt.plot(rewards)
    plt.show()