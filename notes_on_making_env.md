I wrapped the simulation into a gym environment. Once you follow the exact structure and don't ask too many questions things actually work out. 
These are some of the resources I've found useful:

1. https://www.datamachinist.com/creating-virtual-environments-for-reinforcement-learning/part-1-registering-a-custom-gym-environment/
2. https://stackoverflow.com/questions/54259338/how-to-pass-arguments-to-openai-gym-environments-upon-init
3. https://github.com/MattChanTK/gym-maze/blob/master/gym_maze/__init__.py
4. https://github.com/openai/gym-soccer
5. https://www.datahubbs.com/building-custom-gym-environments-for-rl/
6. https://towardsdatascience.com/ray-and-rllib-for-fast-and-parallel-reinforcement-learning-6d31ee21c96c


The process so far: 
1. cd env 
2. pip install . (-e didn't help)
3. python

    3.1 import gym 
    3.2 import env 
    3.3 env = gym.make('rl-v0', env_config={})   

# Register env as a Ray env
    >>> env_1 = gym.make('single-v0', config={'corridor_length' : 10 })

def env_creator(config):
    return gym.make('rl-v0', config)
config = {}
 https://docs.ray.io/en/master/_modules/ray/tune/registry.html
    
    from ray.tune.registry import register_env

    def env_creator(config): return (gym.make('single-v0', config={'corridor_length' : 10 }))
 
    register_env("my_env", env_creator)

    ray.init()
    
    trainer = ppo.PPOTrainer(env="my_env")
    
    from env.envs.simple_single_agent_env import SimpleCorridor
    trainer = ppo.PPOTrainer(env=SimpleCorridor, config={'env_config':{'corridor_length':3}})
    
Run a simple model 
    for i in range(10):
        result = trainer.train()