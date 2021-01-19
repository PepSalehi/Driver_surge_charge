#p.82, Learning Python Application Development
import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path)

from env.envs.MARL_env import MultiAgentEnv

from env.envs.simple_single_agent_env import SimpleCorridor


# optionally print the sys.path for debugging)
#print("in __init__.py sys.path:\n ",sys.path)