from gym.envs.registration import register


register(
    id='single-v0',
    entry_point='env.envs:SimpleCorridor'
)
register(
    id='rl-v0',
    entry_point='env.envs:MultiAgentEnv'
)

