import numpy as np
import gym
from gym import spaces
from lib.Data import Data
from lib.configs import config_dict
from lib.utils import Model
from lib.Constants import POLICY_UPDATE_INTERVAL, WARMUP_TIME_SECONDS, T_TOTAL_SECONDS, INT_ASSIGN, \
    ANALYSIS_TIME_SECONDS, DEMAND_UPDATE_INTERVAL

import time
import os


class MultiAgentEnv(gym.Env):

    def __init__(self, env_config):
        # data_instance = Data.init_from_config_dic(config_dict)
        # self.model = Model(data_instance)
        # self.T = WARMUP_TIME_SECONDS
        self.min_action = 0
        self.max_action = config_dict["MAX_BONUS"]

        # define action/state space per zone
        # https://github.com/openai/gym/wiki/Table-of-environments
        # https://github.com/openai/gym/blob/master/gym/envs/classic_control/continuous_mountain_car.py
        # https://github.com/openai/multiagent-particle-envs/blob/master/bin/interactive.py
        self.action_space = spaces.Box(
            low=self.min_action,
            high=self.max_action,
            shape=(1,),
            dtype=np.float32
        )

        self.reset()

    def step(self, action_n):
        """
        for z in zones: set the bonus value
        action_n is {z_id:bonus}
        instead of the operator, the env sets the bonus values
        """
        for zone in self.model.zones:
            requested_bonus = action_n[zone.id]
            # make sure it is within bounds
            requested_bonus = min(max(requested_bonus, self.min_action), self.max_action)
            if requested_bonus <= self.model.budget:
                self.model.budget -= requested_bonus
                zone.set_bonus(requested_bonus)

        # simulate for a while (e.g., 10 mins)
        for t in range(self.T, self.T + POLICY_UPDATE_INTERVAL, INT_ASSIGN):
            self.model.dispatch_at_time(t)

        # observe the next state
        states = self._get_states()
        # observe the (global) reward
        reward = self._get_reward_unserved()
        # other info
        info = self._get_info()
        # update the clock
        self.T += POLICY_UPDATE_INTERVAL
        # done flag
        done = self._is_done()

        return states, reward, done, info

    def reset(self):
        data_instance = Data.init_from_config_dic(config_dict)
        self.model = Model(data_instance)
        self.T = WARMUP_TIME_SECONDS
        # run the warm up period
        for t in range(self.T, self.T + 3600, INT_ASSIGN):
            self.model.dispatch_at_time(t)
        self.T = ANALYSIS_TIME_SECONDS
        print("##########################")
        print("##########################")
        print("End of the warm up time ")
        print("##########################")
        print("##########################")


    def _get_states(self):
        """
        Place holder. should return info per zone

        :return:
        """
        demand, supply = self.model.get_both_supply_and_demand_per_zone(self.T)
        return demand

    def _get_stats(self):
        self.stats = [z.generate_performance_stats() for z in self.model.zones]

    def _get_reward(self):
        return np.sum([z.reward_dict[np.ceil(self.T / POLICY_UPDATE_INTERVAL)] for z in self.model.zones])

    def _get_reward_unserved(self):
        return np.sum([z.generate_performance_stats(self.T)[3] for z in self.model.zones])

    def _get_info(self):
        return None

    def render(self, mode='human'):
        pass

    def _is_done(self):
        return self.T == T_TOTAL_SECONDS



if __name__ == "__main__":
    MultiAgentEnv()
