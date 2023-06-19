from rlberry.agents.experimental.torch import SACAgent
#from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs import Pendulum

from rlberry.manager import AgentManager

import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
## Training CODE
def env_ctor(env, wrap_spaces=True):
    #return Wrapper(env, wrap_spaces)
    return env

env = gym.make('Hopper-v2')
#env = Pendulum()
#env = gym.wrappers.TimeLimit(env, max_episode_steps=200)


env = gym.wrappers.RecordEpisodeStatistics(env)


env_kwargs = dict(env=env)
agent = AgentManager(
    SACAgent,
    (env_ctor, env_kwargs),
    fit_budget = int(1e6), # 500 episode ~ 100k steps
    n_fit=1,
    enable_tensorboard=True,
    agent_name="Test"
)
agent.fit()