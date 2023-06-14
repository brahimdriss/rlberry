from rlberry.agents.experimental.torch import SACAgent
#from rlberry.envs.benchmarks.ball_exploration import PBall2D
from rlberry.envs import Pendulum

from rlberry.manager import AgentManager

def env_ctor(env, wrap_spaces=True):
    #return Wrapper(env, wrap_spaces)
    return env

env = Pendulum()
n_steps = 1e3

env_kwargs = dict(env=env)
agent = AgentManager(
    SACAgent,
    (env_ctor, env_kwargs),
    fit_budget = 500,
    n_fit=1,
    enable_tensorboard=True,
)

agent.fit()

#agent = SACAgent(env)
#agent.fit(budget=n_steps)

# env.enable_rendering()
# state = env.reset()
# for tt in range(200):
#     action, _, _ = agent.policy(state)
#     action = action.detach().cpu().numpy()
#     next_state, reward, done, _ = env.step(action)
#     state = next_state