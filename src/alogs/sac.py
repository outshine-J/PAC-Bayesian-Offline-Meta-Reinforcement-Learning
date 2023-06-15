import argparse
import os
import time
import warnings
from collections import namedtuple
from copy import deepcopy
from functools import reduce
from operator import mul

import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal

from algos.base import BasePolicy
from common import create_env, set_random_seed
from exploration.random import GaussianNoise
from models.actor import ActorSAC
from models.critic import Critic
from replay_buffer.buffer import CReplay_buffer


class SACPolicy(BasePolicy):
    """
    Implementation of Soft Actor-Critic
    """
    def __init__(self,
                 prm,
                 actor_model,
                 critic1_model,
                 critic2_model,
                 task,
                 observation_space,
                 action_space,
                 tau=0.005,
                 gamma=0.99,
                 alpha=0.2,
                 reward_normalization=False,
                 exploration_noise=GaussianNoise(sigma=0.1),
                 deterministic_eval=True,
                 estimation_step=1,
                 action_scaling=False,
                 action_bound_method="tanh",
                 optimizer=torch.optim.Adam
                 ):
        super(SACPolicy, self).__init__(observation_space=observation_space,
                                        action_space=action_space,
                                        action_scaling=action_scaling,
                                        action_bound_method=action_bound_method)

        # assert action_bound_method != "tanh", "tanh mapping is not supported"

        self.train = True

        self.prm = prm
        self.device = prm.device

        ### Set some parameters
        self.tau = tau
        self._gamma = gamma
        self._noise = exploration_noise
        self._rew_norm = reward_normalization

        # set environment-parameters
        self.env = create_env(prm)
        self.env.reset_task(task)
        # self.env.set_task(task)

        # alpha auto update
        self._is_auto_alpha = prm.is_auto_alpha
        if self._is_auto_alpha:
            self._log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self._alpha = self._log_alpha.exp()
            self._alpha_optim = optimizer([self._log_alpha], lr=prm.lr, eps=1e-4)
            self._target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        else:
            self._alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)

        self._deterministic_eval = deterministic_eval
        self.__eps = np.finfo(np.float32).eps.item()

        ### Create replay_buffer
        self.replay_buffer = CReplay_buffer(prm.replay_size, self.device)

        ### Set network structure
        self.policy_net = actor_model
        self.actor_optim = optimizer(self.policy_net.parameters(), lr=prm.lr)

        self.critic1, self.target_critic1 = critic1_model, deepcopy(critic1_model)
        self.critic2, self.target_critic2 = critic2_model, deepcopy(critic2_model)
        self.critic1_optim, self.critic2_optim = optimizer(self.critic1.parameters(), lr=prm.lr), optimizer(self.critic2.parameters(), lr=prm.lr)

        self.loss_fnc = nn.MSELoss()

    def select_action(self, state):
        """
            Select an action
        """
        state = torch.FloatTensor(state).to(self.device)
        mu, log_sigma = self.policy_net(state)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        z = dist.rsample()      #rsample means it is sampled using reparameterisation trick
        action = torch.tanh(z).detach().cpu().numpy()
        if self._deterministic_eval and not self.train:
            action = torch.tanh(mu).detach().cpu().numpy()
        return action

    def get_action_log_prob(self, state):
        """
            Input a state and Output action and log_prob
        """
        batch_mu, batch_log_sigma = self.policy_net(state)
        batch_sigma = torch.exp(batch_log_sigma)
        dist = Normal(batch_mu, batch_sigma)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = (dist.log_prob(z) - torch.log(1 - action.pow(2) + self.__eps)).sum(1, keepdim=True)
        # print(log_prob.mean().item())

        return action, log_prob

    def run_episode(self):
        '''
            This function add transition to replay buffer
        '''
        episode_timesteps = 0
        episode_reward = 0
        reward_epinfos = []

        state = self.env.reset()
        done = False
        step = 0

        while not done and step < self.prm.max_path_length:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            episode_reward += reward
            reward_epinfos.append(reward)
            episode_timesteps += 1
            step += 1

            # Clip reward to [-1.0, 1.0]
            if hasattr(self.prm, 'clip_reward') and self.prm.clip_reward:
                reward = max(min(reward, 1.0), -1.0)

            self.replay_buffer.push([state, action, reward, next_state, np.array([1 - done])])
            state = next_state

        # Record current episode information
        info = {}
        info['episode_timesteps'] = episode_timesteps
        info['episode_reward'] = episode_reward

        return info

    def batch_sample(self):
        # sample replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.prm.batch_size)
        return [states, actions, rewards, next_states, dones]

    def calculate_critic_losses(self, batch):
        """
            Calculates the losses for the two critics.
            This is the ordinary Q-learning loss except the additional entropy term is taken into account
        """
        states, actions, rewards, next_states, marks = batch
        with torch.no_grad():
            next_actions, actions_log_probs = self.get_action_log_prob(next_states)
            qf1_next_target = self.target_critic1(torch.cat((next_states, next_actions), 1))
            qf2_next_target = self.target_critic2(torch.cat((next_states, next_actions), 1))
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self._alpha * actions_log_probs
            next_q_value = rewards + marks * self._gamma * (min_qf_next_target)
        qf1 = self.critic1(torch.cat((states, actions), 1))
        qf2 = self.critic2(torch.cat((states, actions), 1))
        critic1_loss = self.loss_fnc(qf1, next_q_value)
        critic2_loss = self.loss_fnc(qf2, next_q_value)

        return critic1_loss, critic2_loss

    def calculate_actor_losses(self, batch):
        """
            Calculate the loss for the actor. The loss includes the additional entropy term
        """
        states, actions, rewards, next_states, marks = batch
        new_actions, new_action_log_probs = self.get_action_log_prob(states)
        qf1 = self.critic1(torch.cat((states, new_actions), 1))
        qf2 = self.critic2(torch.cat((states, new_actions), 1))
        min_qf = torch.min(qf1, qf2)
        policy_loss = ((self._alpha * new_action_log_probs) - min_qf).mean()

        return policy_loss, new_action_log_probs

    def calculate_entropy_losses(self, entropies):
        alpha_loss = -torch.mean(self._log_alpha * (self._target_entropy + entropies).detach())
        return alpha_loss

    def do_train(self):
        """
            The function is only used as an adaptive method
        """
        # sample replay buffer
        batch = self.replay_buffer.sample(self.prm.batch_size)
        # Update Critic by one step of gradient descent
        critic1_loss, critic2_loss = self.calculate_critic_losses(batch)
        # Update policy by one step of gradient ascent
        actor_loss, entropies = self.calculate_actor_losses(batch)

        grad_step(actor_loss, self.actor_optim)
        grad_step(critic1_loss, self.critic1_optim)
        grad_step(critic2_loss, self.critic2_optim)

        # Update temperture by one step of gradient gradient
        if self._is_auto_alpha:
            entropy_loss = self.calculate_entropy_losses(entropies)
            grad_step(entropy_loss, self._alpha_optim)
            self._alpha = self._log_alpha.exp()

        # Update target networks
        self.sync_weight()

        return critic1_loss

    def adaption(self, actor_complex_term, critic1_complex_term, critic2_complex_term):

        avg_actor_empiric_loss, avg_critic1_empiric_loss, avg_critic2_empiric_loss = 0, 0, 0
        for i_MC in range(self.prm.n_MC):
            batch = self.replay_buffer.sample(self.prm.batch_size)
            critic1_loss, critic2_loss = self.calculate_critic_losses(batch)
            actor_loss, entropies = self.calculate_actor_losses(batch)

            avg_actor_empiric_loss += actor_loss
            avg_critic1_empiric_loss += critic1_loss
            avg_critic2_empiric_loss += critic2_loss

        avg_actor_empiric_loss /= self.prm.n_MC
        avg_critic1_empiric_loss /= self.prm.n_MC
        avg_critic2_empiric_loss /= self.prm.n_MC

        total_actor_objective = avg_actor_empiric_loss + actor_complex_term
        total_critic1_objective = avg_critic1_empiric_loss + critic1_complex_term
        total_critic2_objective = avg_critic2_empiric_loss + critic2_complex_term

        grad_step(total_actor_objective, self.actor_optim)
        grad_step(total_critic1_objective, self.critic1_optim)
        grad_step(total_critic2_objective, self.critic2_optim)

        if self._is_auto_alpha:
            entropy_loss = self.calculate_entropy_losses(entropies)
            grad_step(entropy_loss, self._alpha_optim)
            self._alpha = self._log_alpha.exp()

        # Update target networks
        self.sync_weight()

    def update_temperture(self, entropy_loss):
        if self._is_auto_alpha:
            grad_step(entropy_loss, self._alpha_optim)
            self._alpha = self._log_alpha.exp()

def grad_step(objective, optimizer):

    optimizer.zero_grad()
    objective.backward()
    optimizer.step()

if __name__ == "__main__":

    import gym

    def sample_tasks(prm, env, num_tasks, train_mode=True):

        return env.unwrapped.sample_tasks(num_tasks)

    parser = argparse.ArgumentParser()

    # optional parameters
    parser.add_argument('--is_auto_alpha', default=True, type=bool)
    parser.add_argument('--iteration', default=2000, type=int)
    parser.add_argument('--total_timesteps', default=0.4e6, type=int)   # TODO

    # Run Parameters
    parser.add_argument('--seed', type=int, help='random seed', default=1)
    parser.add_argument('--gpu_index', type=int, help='The index of GPU device to run on', default=0)

    # Env Parameters
    parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)', default='Pendulum-v0 or MountainCarContinuous-v0 or ant-dir')
    parser.add_argument('--env_name', type=str, help='PuckWorld-v0 or Krazygame-v0 or Maze-v0', default='point-robot-wind')
    parser.add_argument('--max_path_length', type=int, help="Maximum path length per episode", default=200)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=3e-4)
    parser.add_argument('--critic_hidden_sizes', type=tuple, help="The sizes of model's hidden_layers", default=(400, 400))
    parser.add_argument('--actor_hidden_sizes', type=tuple, help="The sizes of model's hidden_layers", default=(400, 400))
    parser.add_argument('--replay_size', type=int, default=int(1e6), help='Replay buffer size int(2000)')
    parser.add_argument('--batch_size', default=250, type=int)  # mini batch size
    parser.add_argument('--num_initial_step', type=int, help="Initial experience pool step size", default=10000)
    parser.add_argument('--gamma', type=float, help='discount factor', default=0.99)
    parser.add_argument('--tau', default=0.005, type=float)  # target smoothing coefficient
    parser.add_argument('--alpha', type=float, default=0.2)  ## Priority

    parser.add_argument('--n_tasks', type=int, default=2)
    parser.add_argument('--randomize_tasks', type=bool, default=True)
    parser.add_argument('--forward_backward', type=bool, default=True)

    # More Parameters
    prm = parser.parse_args()
    prm.device = torch.device("cuda:" + str(prm.gpu_index) if torch.cuda.is_available() else "cpu")

    prm.log_var_init = {'mean': -10.0, 'std': 0.1}

    set_random_seed(prm.seed)

    # Create the environment
    env = create_env(prm)
    env.close()
    prm.env = env
    tasks = env.get_all_task_idx()[:]

    prm.optim_func, prm.optim_args = optim.Adam, {'lr': prm.lr}
    prm.lr_schedule = {}

    for task in tasks:
        env.reset_task(task)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        actor_model = ActorSAC(state_dim=state_dim, action_dim=action_dim, hidden_sizes=prm.actor_hidden_sizes).to(prm.device)

        critic1_model = Critic(input_dim=state_dim + action_dim, output_dim=1, hidden_sizes=prm.critic_hidden_sizes).to(prm.device)
        critic2_model = Critic(input_dim=state_dim + action_dim, output_dim=1, hidden_sizes=prm.critic_hidden_sizes).to(prm.device)

        agent = SACPolicy(prm=prm, action_space=env.action_space, observation_space=env.observation_space,
                                  task=task, actor_model=actor_model, critic1_model=critic1_model, critic2_model=critic2_model,
                                  tau=prm.tau)

        episode_nums = []
        episode_times = []
        episode_rewards = []
        episode_losses = []
        do_train = 0
        total_timesteps = 0
        i = 0

        while total_timesteps < prm.total_timesteps:

            i += 1
            episode_timesteps = 0
            episode_reward = 0
            state = env.reset()
            done = False
            step = 0
            loss = torch.zeros(1)

            while not done and step <= prm.max_path_length:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                env.render()
                os.system("pause")
                # time.sleep(1000000)


                if step == prm.max_path_length-1:
                    masked_done = False
                else:
                    masked_done = done
                agent.replay_buffer.push([state, action, reward, next_state, np.array([1-masked_done])])

                state = next_state
                # agent.env.render()

                episode_reward += reward
                episode_timesteps += 1
                step += 1

                if agent.replay_buffer.size_rb() >= prm.num_initial_step:
                    loss = agent.do_train()
                    do_train += 1
                # if do_train > 200:
                #     agent.reset_model()
                #     do_train %= 200
                if done or step >= prm.max_path_length:
                    if i % 1 == 0:
                        # print("Ep_i {}, the ep_r is {}, the t is {}, do_train is {}, loss is {}".format(i, episode_reward, step, do_train, loss.item()))
                        print("Ep_i {}, the ep_r is {}, the t is {}, the total_steps is {}, loss is {:.4}".format(i, episode_reward, step, total_timesteps, loss.item()))
                    break
            env.close()

            total_timesteps += episode_timesteps

            episode_nums.append(i+1)
            episode_losses.append(loss.detach().cpu().numpy())
            episode_times.append(total_timesteps)
            episode_rewards.append(episode_reward)

        def plot(timesteps, return_reward, critic_loss):
            import matplotlib.pyplot as plt

            plt.plot(timesteps, return_reward)
            plt.plot(timesteps, critic_loss)
            # plt.xlabel("gamma = " + str(prm.gamma))
            plt.xlabel("actor_hidden_sizes = " + str(prm.actor_hidden_sizes) + " batch_size = " + str(prm.batch_size))
            plt.show()

        # plot(episode_nums, episode_rewards, episode_losses)
        plot(episode_times, episode_rewards, episode_losses)

        num_rollout = 50
        data_dir = "../data/goal_idx{}".format(task)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        for ntrj in range(num_rollout):
            state = env.reset()
            done = False
            episode_reward = 0
            trj = []
            step = 0
            while not done and step < prm.max_path_length:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                trj.append([state, action, reward, next_state])
                state = next_state
                episode_reward += reward
                step += 1
            np.save(os.path.join(data_dir, f'trj_evalsample{ntrj}_step800000.npy'), np.array(trj))

