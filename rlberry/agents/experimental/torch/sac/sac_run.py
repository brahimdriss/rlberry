import time

import gym.spaces as spaces
import numpy as np
import rlberry
import torch
import torch.nn as nn
import torch.optim as optim
from rlberry.agents import AgentWithSimplePolicy
from rlberry.agents.torch.utils.training import (model_factory,
                                                 optimizer_factory)
from rlberry.agents.utils.replay import ReplayBuffer
from rlberry.utils.torch import choose_device
from rlberry.utils.factory import load

logger = rlberry.logger


def default_q_net_fn(env, **kwargs):
    """
    Returns a default Q value network.
    """
    del kwargs
    model_config = {
        "type": "MultiLayerPerceptron",
        "layer_sizes": (256, 256),
        "reshape": True,
        "in_size": env.observation_space.shape[0] + env.action_space.shape[0],
        "out_size": 1,
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
            "reshape": True,
            "is_policy": True,
            "ctns_actions": True,
            "sac": True
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
        Online model with continuous (Box) state space and continuous actions
    batch_size : int
        Number of episodes to wait before updating the policy.
    gamma : double
        Discount factor in [0, 1].
    learning_rate : double
        Learning rate.
    buffer_capacity : int
        Capacity of the replay buffer 
    optimizer_type: str
        Type of optimizer. 'ADAM' by defaut.
    tau : double
        Target smoothing coefficient
    policy frequency
        Policy training frequency (Delayed TD3 update)
    alpha
        Entropy regularization coefficient
    autotunealpha
        Automatic tuning of alpha
    learning start
        Timesteps done before training starts
    policy_net_fn : function(env, **kwargs)
        Function that returns an instance of a policy network (pytorch).
        If None, a default net is used.
    policy_net_kwargs : dict
        kwargs for policy_net_fn
    q_net_constructor : Callable, str or None
        Function/constructor that returns a torch module for the Q-network
    q_net_kwargs : optional, dict
        Parameters for q_net_constructor.
    device : str
        Device to put the tensors on
    writer_frequency : int
        Frequency of tensorboard logging 
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
        autotune_alpha=True,
        learning_start=5e3,
        policy_net_fn=None,
        policy_net_kwargs=None,
        q_net_constructor=None,
        q_net_kwargs=None,
        writer_frequency = 100,
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
        self.device = choose_device(device)
        self.writer_frequency = writer_frequency

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
        self.action_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0,
                                          dtype=torch.float32).to(self.device)
        self.action_bias = torch.tensor((self.env.action_space.high + self.env.action_space.low) / 2.0,
                                         dtype=torch.float32).to(self.device)        
        
        # Autotune alpha or use a fixed default value
        self.autotune = autotune_alpha
        if not self.autotune:
            self.alpha = alpha

        # initialize
        self.reset()


    def reset(self, **kwargs):
        """
        Reset the agent.
        This function resets the agent by initializing the necessary components and parameters for training.
        
        Args:
            **kwargs: Additional keyword arguments.
        """

        # Initialize the rollout buffer
        self.memory = ReplayBuffer(max_replay_size=self.buffer_capacity, rng=self.rng)
        self.memory.setup_entry("states", dtype=np.float32)
        self.memory.setup_entry("next_states", dtype=np.float32)
        self.memory.setup_entry("actions", dtype=np.float32)
        self.memory.setup_entry("rewards", dtype=np.float32)
        self.memory.setup_entry("dones", dtype=np.float32)

        # Intialize the Actor
        self.cont_policy = self.policy_net_fn(self.env, **self.policy_net_kwargs).to(self.device)
        self.policy_optimizer = optimizer_factory(self.cont_policy.parameters(), **self.policy_optimizer_kwargs)
        self.cont_policy.load_state_dict(self.cont_policy.state_dict())

        # Intialize the double Q networks and the target Q networks
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
        
        # Define the loss
        self.MseLoss = nn.MSELoss()
        
        # Automatic entropy tuning
        if self.autotune:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=self.q_learning_rate)
        
        # initialize episode, steps and time counters
        self.total_episodes = 0
        self.total_timesteps = 0
        self.time = time.time()

    def policy(self):
        pass

    def fit(self, budget: int, **kwargs):
        """
        Train the agent using the provided environment.

        Parameters
        ----------
        budget: int
            number of episodes. Each episode runs for self.horizon unless it
            enconters a terminal state in which case it stops early.
        """
        state = self.env.reset()

        while self.total_timesteps < budget:
            if self.total_timesteps < self.learning_start:
                action = np.array(self.env.action_space.sample())
            else:                
                tensor_state = np.array([state])
                action, _ = self._select_action(tensor_state)
                action = action.detach().cpu().numpy()[0]
            
            next_state, reward, done, info = self.env.step(action)
            

            # TODO : Try With and without real next state
            real_next_state = next_state.copy()

            if done and "terminal_observation" in info.keys():
                real_next_state = info["terminal_observation"]

            if "episode" in info.keys():
                self.writer.add_scalar("episode/episode_rewards", info["episode"]["r"], self.total_timesteps)
                self.writer.add_scalar("episode/episode_length", info["episode"]["l"], self.total_timesteps)


            # Add to replay buffer
            self.memory.append(
                {
                    "states": state,
                    "next_states": next_state,
                    "actions": action,
                    "rewards": reward,
                    "dones": done,
                }
            )

            # update state
            state = next_state

            # TODO : Try With and without end episode
            if done:
                state = self.env.reset()
                self.memory.end_episode()
            
            if self.total_timesteps > self.learning_start:
                self._update()

            self.total_timesteps += 1

        print(f"Timesteps done : {self.total_timesteps}")


    def _select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_dist = self.cont_policy(state)
        mean, log_std = action_dist
        std = log_std.exp()
        action_dist = torch.distributions.Normal(mean, std)
        # Reparametrization trick
        x_t = action_dist.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = action_dist.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


    def _update(self):
        memory_data = self.memory.sample(self.batch_size, 1).data
        states = torch.tensor(memory_data["states"]).view(self.batch_size,-1).to(self.device)
        next_state = torch.tensor(memory_data["next_states"]).view(self.batch_size,-1).to(self.device)
        actions = torch.tensor(memory_data["actions"]).view(self.batch_size,-1).to(self.device)
        rewards = torch.tensor(memory_data["rewards"]).view(self.batch_size,-1).to(self.device)
        dones = torch.tensor(memory_data["dones"]).view(self.batch_size,-1).to(self.device)

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


        # Actor update
        act_loss = None
        alpha_loss = None
        state_log_pi = None
        if self.total_timesteps % self.policy_frequency == 0 :    #TD3 Delayed update
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
        

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


        if self.writer is not None and self.total_episodes % self.writer_frequency == 0:
            self.writer.add_scalar(
                "fit/loss_q1", float(q1_loss_v.detach()), self.total_timesteps
            )
            self.writer.add_scalar(
                "fit/loss_q2", float(q2_loss_v.detach()), self.total_timesteps
            )
            self.writer.add_scalar(
                "fit/value_q1", float(q1_v.mean().detach()), self.total_timesteps
            )
            self.writer.add_scalar(
                "fit/value_q2", float(q2_v.mean().detach()), self.total_timesteps
            )
            if act_loss:    
                self.writer.add_scalar(
                "fit/loss_act", float(act_loss.detach()), self.total_timesteps
            )
                self.writer.add_scalar(
                "fit/alpha", float(self.alpha), self.total_timesteps
            )
            self.writer.add_scalar(
                "episode/SPS", int(self.total_timesteps / (time.time() - self.time)), self.total_timesteps
        )
            if self.autotune and alpha_loss:
                self.writer.add_scalar(
                "fit/alpha_loss", float(alpha_loss.detach()), self.total_timesteps
            )      

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
