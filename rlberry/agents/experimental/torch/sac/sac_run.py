from rlberry.agents.experimental.torch.sac.utils import ReplayBuffer, get_qref, get_vref, alpha_sync
import time

import torch
import torch.nn as nn
from torch.nn.functional import one_hot
import numpy as np

import gym.spaces as spaces

from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.torch.utils.training import optimizer_factory
from rlberry.agents.torch.utils.training import model_factory
from rlberry.agents.torch.utils.models import default_policy_net_fn
from rlberry.agents.torch.utils.models import default_value_net_fn
from rlberry.agents.torch.utils.models import default_twinq_net_fn
from rlberry.utils.torch import choose_device
from rlberry.wrappers.uncertainty_estimator_wrapper import UncertaintyEstimatorWrapper

from stable_baselines3.common.buffers import ReplayBuffer

import rlberry
import torch.optim as optim

logger = rlberry.logger

def default_q_net_fn(env, **kwargs):
    """
    Returns a default Q value network.
    """
    del kwargs
    model_config = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (256, 256),
        "reshape": False,
        "in_size": env.observation_space.shape[0] + env.action_space.shape[0],
        "out_size": 1

    }
    return model_factory(**model_config)

def default_policy_net_fn(env, **kwargs):
    """
    Returns a default Q value network.
    """
    del kwargs
    model_config = {
            "type": "MultiLayerPerceptron",
            "in_size": env.observation_space.shape[0],
            "layer_sizes": [256, 256],
            "out_size": env.action_space.shape[0],
            "reshape": False,
            "is_policy": True,
            "ctns_actions": True
    }
    return model_factory(**model_config)


class SACAgent(AgentWithSimplePolicy):
    """
    Experimental Soft Actor Critic Agent (WIP).

    SAC, or SOFT Actor Critic, an offpolicy actor-critic deep RL algorithm
    based on the maximum entropy reinforcement learning framework. In this
    framework, the actor aims to maximize expected reward while also
    maximizing entropy.

    Parameters
    ----------
    env : Model
        Online model with continuous (Box) state space and discrete actions
    batch_size : int
        Number of episodes to wait before updating the policy.
    gamma : double
        Discount factor in [0, 1].
    entr_coef : double
        Entropy coefficient.
    learning_rate : double
        Learning rate.
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    value_net_fn : function(env, **kwargs)
        Function that returns an instance of a value network (pytorch).
        If None, a default net is used.
    twinq_net_fn : function(env, **kwargs)
        Function that returns a tuple composed of two Q networks (pytorch).
        If None, a default net function is used.
    policy_net_kwargs : dict
        kwargs for policy_net_fn
    value_net_kwargs : dict
        kwargs for value_net_fn
    q_net_kwargs : dict
        kwargs for q_net_fn
    use_bonus : bool, default = False
        If true, compute an 'exploration_bonus' and add it to the reward.
        See also UncertaintyEstimatorWrapper.
    uncertainty_estimator_kwargs : dict
        Arguments for the UncertaintyEstimatorWrapper
    device : str
        Device to put the tensors on

    References
    ----------
    Haarnoja, Tuomas, et al. "Soft actor-critic algorithms and applications."
    arXiv preprint arXiv:1812.05905 (2018).
    """

    name = "SAC"

    def __init__(
        self,
        env,
        batch_size=256,
        gamma=0.99,
        q_learning_rate=1e-3,
        policy_learning_rate=3e-4,
        buffer_capacity: int = int(1e6),
        optimizer_type="ADAM",
        tau=0.005,
        policy_frequency=2,
        alpha=0.2,
        autotune_alpha=False,
        learning_start=5e3,
        writer_frequency=10,
        policy_net_fn=None,
        policy_net_kwargs=None,
        q_net_constructor=None,
        q_net_kwargs=None,
        use_bonus=False,
        uncertainty_estimator_kwargs=None,
        device="cuda:best",
        **kwargs
    ):
        AgentWithSimplePolicy.__init__(self, env, **kwargs)

        # check environment
        assert isinstance(self.env.observation_space, spaces.Box)
        assert isinstance(self.env.action_space, spaces.Box)

        self.batch_size = batch_size
        self.gamma = gamma
        self.q_learning_rate = q_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.buffer_capacity = buffer_capacity
        self.learning_start = learning_start
        self.policy_frequency = policy_frequency
        self.tau = tau
        self.writer_frequency = writer_frequency
        self.device = choose_device(device)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        
        self.policy_net_kwargs = policy_net_kwargs or {}

        if isinstance(q_net_constructor, str):
            q_net_ctor = load(q_net_constructor)
        elif q_net_constructor is None:
            q_net_ctor = default_q_net_fn
        else:
            q_net_ctor = q_net_constructor
        
        q_net_kwargs = q_net_kwargs or {}
        self.q_net_kwargs = q_net_kwargs
        self.q_net_ctor = q_net_ctor 

        self.q1 = q_net_ctor(self.env, **q_net_kwargs).to(self.device)
        self.q2 = q_net_ctor(self.env, **q_net_kwargs).to(self.device)
        self.q1_target = q_net_ctor(self.env, **q_net_kwargs).to(self.device)
        self.q2_target = q_net_ctor(self.env, **q_net_kwargs).to(self.device)


        self.policy_net_fn = policy_net_fn or default_policy_net_fn

        self.q_optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": q_learning_rate}
        self.policy_optimizer_kwargs = {"optimizer_type": optimizer_type, "lr": policy_learning_rate}
        
        self.action_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)
        self.action_bias = torch.tensor((self.env.action_space.high + self.env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)        
        

        self.autotune = autotune_alpha
        if not self.autotune:
            self.alpha = alpha

        # initialize
        self.reset()

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(
        self.batch_size,
        self.env.observation_space,
        self.env.action_space,
        handle_timeout_termination=True,
    )
        

    def reset(self, **kwargs):
        # actor
        self.cont_policy = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(self.device)
        self.policy_optimizer = optimizer_factory(self.cont_policy.parameters(), **self.policy_optimizer_kwargs)
        self.cont_policy.load_state_dict(self.cont_policy.state_dict())

        ## TODO : Load state dict for Q1/Q2 and targets ?
        self.q1 = self.q_net_ctor(self.env, **self.q_net_kwargs)
        self.q2 = self.q_net_ctor(self.env, **self.q_net_kwargs)
        self.q1_target = self.q_net_ctor(self.env, **self.q_net_kwargs)
        self.q2_target = self.q_net_ctor(self.env, **self.q_net_kwargs)
        self.q1.to(self.device)
        self.q2.to(self.device)
        self.q1_target.to(self.device)
        self.q2_target.to(self.device)
        self.q1_optimizer = optimizer_factory(
            self.q1.parameters(), **self.q_optimizer_kwargs
        )
        self.q2_optimizer = optimizer_factory(
            self.q2.parameters(), **self.q_optimizer_kwargs
        )
        self.q1_target_optimizer = optimizer_factory(
            self.q1.parameters(), **self.q_optimizer_kwargs
        )
        self.q2_target_optimizer = optimizer_factory(
            self.q2.parameters(), **self.q_optimizer_kwargs
        )

        self.MseLoss = nn.MSELoss()
        
        # Automatic entropy tuning
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_learning_rate)
        
        # initialize episode, steps and time counters
        self.episode = 0
        self.steps = 0
        self.time = time.time()

    def policy(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_dist = self.cont_policy(state)
        # Std squash
        mean, std = action_dist.mean, action_dist.stddev
        log_std = torch.tanh(torch.log(std))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        action_dist = torch.distributions.Normal(mean, std)
        
        x_t = action_dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = action_dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            number of episodes. Each episode runs for self.horizon unless it
            enconters a terminal state in which case it stops early.
        """
        del kwargs
        n_steps_to_run = budget
        count = 0
        while count < n_steps_to_run:
            self._run_episode()
            count += 1
        print(f"Timesteps done : {self.steps}")

    def _get_batch(self, device="cpu"):
        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_action_log_prob,
            batch_reward,
            batch_done,
        ) = self.replay_buffer.sample(self.batch_size)

        # convert to torch tensors
        batch_state_tensor = torch.FloatTensor(batch_state).to(self.device)
        batch_next_state_tensor = torch.FloatTensor(batch_next_state).to(self.device)
        batch_action_tensor = torch.LongTensor(batch_action).to(self.device)
        batch_action_log_prob_tensor = torch.FloatTensor(batch_action_log_prob).to(
            self.device
        )
        batch_reward_tensor = (
            torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        )
        batch_done_tensor = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)

        return (
            batch_state_tensor,
            batch_next_state_tensor,
            batch_action_tensor,
            batch_action_log_prob_tensor,
            batch_reward_tensor,
            batch_done_tensor,
        )

    def _select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_dist = self.cont_policy(state)
        # Std squash
        mean, std = action_dist.mean, action_dist.stddev
        log_std = torch.tanh(torch.log(std))
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()
        action_dist = torch.distributions.Normal(mean, std)
        
        x_t = action_dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = action_dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


    def _run_episode(self):
        """
        """
        # interact for H steps
        episode_rewards = 0
        state = self.env.reset()
        done = False
        while not done:
            self.steps += 1
            if self.steps < self.learning_start:
                action = np.array(self.env.action_space.sample())
                
                action_logprob = np.array([[-1]])
            else:                
                tensor_state = np.array([state])
                action, action_logprob = self._select_action(tensor_state)
                action = action.detach().cpu().numpy()[0]
                action_logprob = action_logprob.detach().cpu().numpy()
            
            next_state, reward, done, info = self.env.step(action)
            
            episode_rewards += reward


            # self.replay_buffer.push(
            #     (state, next_state, action, action_logprob, reward, done)
            # )
            self.replay_buffer.add(state, next_state, action, reward, done, [info])


            # update state
            state = next_state
            if self.steps > self.learning_start:
                self._update()

        self.episode += 1

        # add rewards to writer
        if self.writer is not None and self.episode % self.writer_frequency == 0:
            self.writer.add_scalar("episode_rewards", episode_rewards, self.episode)

        return episode_rewards



    def _update(self):
        # sample batch
        # batch = self._get_batch(self.device)
        # states, next_state, actions, _, rewards, dones = batch
        data = self.replay_buffer.sample(self.batch_size)
        states = data.observations.to(self.device)
        next_state = data.next_observations.to(self.device)
        actions = data.actions.to(self.device)
        rewards = data.rewards.to(self.device)
        dones = data.dones.to(self.device)
        
        with torch.no_grad():
                next_state_actions, next_state_log_pi = self._select_action(next_state.detach().cpu().numpy())
                q1_next_target = self.q1_target(torch.cat([next_state, next_state_actions], dim=1))
                q2_next_target = self.q2_target(torch.cat([next_state, next_state_actions], dim=1))
                min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_state_log_pi
                next_q_value = rewards.flatten() + (1 - dones.flatten()) * self.gamma * (min_q_next_target).view(-1)

        q1_v = self.q1(torch.cat([states, actions], dim=1))
        q2_v = self.q2(torch.cat([states, actions], dim=1))
        q1_loss_v = self.MseLoss(q1_v.squeeze(), next_q_value)
        q2_loss_v = self.MseLoss(q2_v.squeeze(), next_q_value)
        q_loss_v = q1_loss_v + q2_loss_v

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()        
        q_loss_v.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        

        # Actor
        if self.steps % self.policy_frequency == 0 :    #TD3 Delayed update
            for _ in range(self.policy_frequency):
                state_action, state_log_pi = self._select_action(states.detach().cpu().numpy())
                q_out_v1 = self.q1(torch.cat([states, state_action], dim=1))
                q_out_v2 = self.q2(torch.cat([states, state_action], dim=1))
                q_out_v = torch.min(q_out_v1, q_out_v2)
                act_loss = ((self.alpha * state_log_pi) - q_out_v).mean()
                self.policy_optimizer.zero_grad()
                act_loss.backward()
                self.policy_optimizer.step()

                if self.autotune:
                    with torch.no_grad():
                        state_action, state_log_pi = self._select_action(states.detach().cpu().numpy())
                    alpha_loss = (-self.log_alpha * (state_log_pi + self.target_entropy)).mean()

                    self.a_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.a_optimizer.step()
                    self.alpha = self.log_alpha.exp().item()

            if self.writer is not None and self.episode % self.writer_frequency == 0:
                self.writer.add_scalar(
                "loss_act", float(act_loss.detach()), self.episode
            )
                self.writer.add_scalar(
                "alpha", float(self.alpha), self.episode
            )
                self.writer.add_scalar("SPS", int(self.steps / (time.time() - self.time)), self.episode)
                if self.autotune:
                    self.writer.add_scalar(
                    "alpha_loss", float(alpha_loss.detach()), self.episode
                )      
        
        if self.writer is not None and self.episode % self.writer_frequency == 0:
            self.writer.add_scalar(
                "loss_q1", float(q1_loss_v.detach()), self.episode
            )
            self.writer.add_scalar(
                "loss_q2", float(q2_loss_v.detach()), self.episode
            )
            self.writer.add_scalar(
                "value_q1", float(q1_v.mean().detach()), self.episode
            )
            self.writer.add_scalar(
                "value_q2", float(q2_v.mean().detach()), self.episode
            )
            self.writer.add_scalar(
                "steps", self.steps, self.episode
            )    

        alpha_sync(self.q1, self.q1_target, 1 - self.tau)
        alpha_sync(self.q2, self.q2_target, 1 - self.tau)


    #
    # For hyperparameter optimization
    #
    @classmethod
    def sample_parameters(cls, trial):
        batch_size = trial.suggest_categorical("batch_size", [1, 4, 8, 16, 32])
        gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.99])
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
        #entr_coef = trial.suggest_loguniform("entr_coef", 1e-8, 0.1)
        #k_epochs = trial.suggest_categorical("k_epochs", [1, 5, 10, 20])

        return {
            "batch_size": batch_size,
            "gamma": gamma,
            "learning_rate": learning_rate,
        }