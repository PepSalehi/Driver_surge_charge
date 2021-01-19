import ray
import gym
from ray.rllib import agents
from ray.rllib.agents import ppo
from env.envs.simple_single_agent_env import SimpleCorridor
from ray.tune.logger import pretty_print


ray.init()
# https://docs.ray.io/en/master/rllib-training.html
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 8
config['env_config'] = {'corridor_length': 10}
config["monitor"] = True
# trainer = ppo.PPOTrainer(env=SimpleCorridor, config={'env_config': {'corridor_length': 10}})
trainer = ppo.PPOTrainer(env=SimpleCorridor, config=config)

for _ in range(30):
    result = trainer.train()
    print("######")
    print("_ is ", _)
    print(pretty_print(result))
    print("######")


ray.shutdown()