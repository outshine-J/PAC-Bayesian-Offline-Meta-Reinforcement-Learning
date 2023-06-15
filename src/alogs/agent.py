import glob
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from src.models.actor import get_actor_model
from src.models.critic import get_critic_model


class TD3_BC(object):
    """
        Implementation of TD3-BC
    """
    def __init__(self,
                 prm,
                 actor_model,
                 critic_model,
                 multi_buffer,
                 env,
                 task,
                 max_action,
                 tau=0.005,
                 gamma=0.2,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 alpha=2.5,
                 optimizer=torch.optim.Adam
                 ):

        self.prm = prm
        self.device = prm.device

        ### Set some parameters
        self.tau = tau
        self.gamma = gamma
        self.alpha = alpha
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = max_action

        # set environment-parameters
        self.env = env
        self.task = task
        self.multi_buffer = multi_buffer
        self.replay_buffer = multi_buffer.task_buffers[task]
        self.normalize_states()

        ### Set network structure
        self.actor_net = actor_model
        self.target_actor = deepcopy(self.actor_net)
        self.actor_optim = optimizer(self.actor_net.parameters(), lr=prm.lr)

        self.critic_net = critic_model
        self.target_critic = deepcopy(self.critic_net)
        self.critic_optim = optimizer(self.critic_net.parameters(), lr=prm.lr)

        self.loss_fnc = nn.MSELoss()

        self.total_it = 0

    def select_action(self, state):
        state = (np.array(state).reshape(1, -1) - self.mean) / self.std
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor_net(state).detach().cpu().numpy().flatten()

    def batch_sample(self):
        # sample replay buffer
        states, actions, rewards, next_states, marks, = self.replay_buffer.random_batch(self.prm.batch_size)
        return [states, actions, rewards, next_states, marks]

    def calculate_critic_losses(self, batch):
        """
            Calculates the losses for the two critics.
            This is the ordinary Q-learning loss except the additional entropy term is taken into account
        """
        states, actions, rewards, next_states, marks = batch

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.target_critic(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (1 - marks) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_net(states, actions)

        # Compute critic loss
        critic_loss = self.loss_fnc(current_Q1, target_Q) + self.loss_fnc(current_Q2, target_Q)

        return critic_loss

    def calculate_actor_losses(self, batch):
        """
            Calculate the loss for the actor. The loss includes the additional entropy term
        """
        states, actions, rewards, next_states, marks = batch

        pi = self.actor_net(states)
        Q = self.critic_net.Q1(states, pi)
        lmbda = self.alpha / Q.abs().mean().detach()

        actor_loss = -lmbda * Q.mean() + self.loss_fnc(pi, actions)

        return actor_loss

    def do_train(self):
        """
            The function is only used as an test method
        """
        self.total_it += 1

        # sample replay buffer
        batch = self.batch_sample()
        # Compute critic loss
        critic_loss = self.calculate_critic_losses(batch)
        # Update Critic by one step of gradient descent
        self.grad_step(critic_loss, self.critic_optim)

        # Delayed policy updates
        if self.total_it % self.prm.policy_freq == 0:
            # Compute actor loss
            actor_loss = self.calculate_actor_losses(batch)
            # Update policy by one step of gradient ascent
            self.grad_step(actor_loss, self.actor_optim)
            # Update the frozen target models
            self.syn_weight()

    def adaptation(self, actor_complex_term=None, critic_complex_term=None, update_policy=False):
        avg_actor_empiric_loss, avg_critic_empiric_loss = 0, 0
        for i_MC in range(self.prm.n_MC):
            batch = self.batch_sample()
            critic_loss = self.calculate_critic_losses(batch)
            if update_policy:
                actor_loss = self.calculate_actor_losses(batch)
                avg_actor_empiric_loss += actor_loss
            avg_critic_empiric_loss += critic_loss

        avg_actor_empiric_loss /= self.prm.n_MC
        avg_critic_empiric_loss /= self.prm.n_MC

        if update_policy:
            total_actor_objective = avg_actor_empiric_loss + actor_complex_term     # TODO
            # total_actor_objective = avg_actor_empiric_loss
            self.grad_step(total_actor_objective, self.actor_optim)
            # Update the frozen target models
            self.syn_weight()

        total_critic_objective = avg_critic_empiric_loss + critic_complex_term      # TODO
        # total_critic_objective = avg_critic_empiric_loss
        self.grad_step(total_critic_objective, self.critic_optim)


    def normalize_states(self):
        if self.prm.normalize:
            self.mean, self.std = self.replay_buffer.normalize_states()
        else:
            self.mean, self.std = 0, 1

    def run_episode(self):

        self.reset_env()

        episode_timesteps = 0
        episode_reward = 0

        state = self.env.reset()
        done = False
        step = 0
        while not done and step < self.prm.max_path_length:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state

            episode_reward += reward
            episode_timesteps += 1
            step += 1

        # Record current episode information
        info = {}
        info['episode_timesteps'] = episode_timesteps
        info['episode_reward'] = episode_reward

        return info

    def empty_buffer(self):
        self.replay_buffer.empty_buffer()

    def syn_weight(self):
        self.soft_update(self.target_critic, self.critic_net, self.tau)
        self.soft_update(self.target_actor, self.actor_net, self.tau)

    def soft_update(self, tgt, src, tau):
        """Softly update the parameters of target module towards the parameters \
        of source module."""
        for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)

    def reset_model(self):
        self.target_actor.load_state_dict(self.actor_net.state_dict())
        self.target_critic.load_state_dict(self.critic_net.state_dict())

    def reset_env(self):
        self.env.reset_task(self.task)

    def grad_step(self, objective, optimizer):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    def add_trajectory(self, trajectory):
        self.replay_buffer.add_trajectory(trajectory)

    def save_model_states(self):
        if not hasattr(self, 'actor_copy'):
            self.actor_copy = deepcopy(self.actor_net)
            self.target_actor_copy = deepcopy(self.target_actor)
            self.critic_copy = deepcopy(self.critic_net)
            self.target_critic_copy = deepcopy(self.target_critic)
        else:
            self.actor_copy.load_state_dict(self.actor_net.state_dict())
            self.target_actor_copy.load_state_dict(self.target_actor.state_dict())
            self.critic_copy.load_state_dict(self.critic_net.state_dict())
            self.target_critic_copy.load_state_dict(self.target_critic.state_dict())

    def rollback(self):
        """
            This function rollback everything to state before test-adaptation
        """
        self.actor_net.load_state_dict(self.actor_copy.state_dict())
        self.target_actor.load_state_dict(self.target_actor_copy.state_dict())
        self.critic_net.load_state_dict(self.critic_copy.state_dict())
        self.target_critic.load_state_dict(self.target_critic_copy.state_dict())


if __name__ == "__main__":

    import d4rl
    import gym
    import argparse
    from src.utils.util import config_tasks_envs
    from src.utils.common import create_env, set_random_seed
    from src.rlkit.envs.wrappers import NormalizedBoxEnv

    from src.rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer


    parser = argparse.ArgumentParser()
    # Add Parameters
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--env_configs', type=str, help='config file path', default='../../configs/ant-dir.json')
    parser.add_argument('--gpu_index', type=int, help='The index of GPU device to run on', default=0)
    # parser.add_argument('--env_name', type=str, help='', default='hopper-medium-v0')
    parser.add_argument('--env_name', type=str, help='', default='ant-dir')
    parser.add_argument('--max_path_length', type=int, help="Maximum path length per episode", default=200)
    parser.add_argument('--n_tasks', type=int, default=2)
    parser.add_argument('--forward_backward', type=bool, default=True)
    parser.add_argument('--n_trj', type=int, default=50)

    # Algo Parameters
    parser.add_argument('--total_timesteps', type=int, default=int(30000))
    parser.add_argument('--eval_freq', type=int, default=int(50))
    parser.add_argument('--expl_noise', default=0.1)
    parser.add_argument('--lr', type=float, help='initial learning rate', default=3e-4)     # TODO
    parser.add_argument('--hidden_sizes', type=tuple, help="The sizes of model's hidden_layers", default=(300, 300, 300))
    parser.add_argument('--replay_size', type=int, default=int(1e4), help='Replay buffer size int(2000)')
    parser.add_argument('--batch_size', type=int, default=300)

    parser.add_argument('--gamma', type=float, help='discount factor', default=0.99)    # TODO
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_noise', type=float, default=0.1)  # TODO
    parser.add_argument('--noise_clip', default=0.5)        # TODO
    parser.add_argument('--policy_freq', type=int, default=2)
    parser.add_argument('--alpha', default=0.1)         # TODO
    parser.add_argument('--normalize', default=True)

    parser.add_argument('--is_load_model', default=True)
    parser.add_argument('--model_dir', default='../../model.pt')

    # More Parameters
    prm = parser.parse_args()
    prm.device = torch.device("cuda:" + str(prm.gpu_index) if torch.cuda.is_available() else "cpu")

    prm.log_var_init = {'mean': -10.0, 'std': 0.1}

    # read tasks/env config params and update args
    config_tasks_envs(prm)

    # Create the environment
    env = create_env(prm)
    env.close()
    prm.env = env
    task = env.get_all_task_idx()[0]
    env.reset_task(task)

    # env = gym.make(prm.env_name)
    # prm.env = env
    # task = 0

    set_random_seed(prm.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Set random seed
    prm.optim_func, prm.optim_args = optim.Adam, {'lr': prm.lr}
    prm.lr_schedule = {}

    # Create multi_buffer
    replay_buffer = MultiTaskReplayBuffer(prm.replay_size, prm.env, [task], 1, prm.device)

    def Init_buffer(prm, task, replay_buffer):
        train_trj_paths = []

        data_dir = '../../data/ant_dir_norm'
        for n in range(prm.n_trj):
            # train_trj_paths += glob.glob(os.path.join(data_dir, "goal_idx%d" %(task), "trj_evalsample%d_step*.npy"%(n)))
            train_trj_paths += glob.glob(os.path.join(data_dir, "goal_idx%d" % (task), "trj_evalsample%d_step800000.npy" % (n)))
            # train_trj_paths += glob.glob(os.path.join(data_dir, "goal_idx%d" %(task), "trj_eval4_step*.npy"))
        train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('\\')[-2].split('goal_idx')[-1]) in [task]]
        train_task_idxs = [int(train_trj_path.split('\\')[-2].split('goal_idx')[-1]) for train_trj_path in
                           train_trj_paths if int(train_trj_path.split('\\')[-2].split('goal_idx')[-1]) in [task]]

        # training task list
        obs_train_lst = []
        action_train_lst = []
        reward_train_lst = []
        next_obs_train_lst = []
        terminal_train_lst = []
        task_train_lst = []

        for train_path, train_task_idx in zip(train_paths, train_task_idxs):
            trj_npy = np.load(train_path, allow_pickle=True)
            obs_train_lst += list(trj_npy[:, 0])
            action_train_lst += list(trj_npy[:, 1])
            reward_train_lst += list(trj_npy[:, 2])
            next_obs_train_lst += list(trj_npy[:, 3])
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_train_lst += terminal
            task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
            task_train_lst += task_train

        for i, (task_train, obs, action, reward, next_obs, terminal) in \
                enumerate(zip(task_train_lst,
                              obs_train_lst,
                              action_train_lst,
                              reward_train_lst,
                              next_obs_train_lst,
                              terminal_train_lst)):
            replay_buffer.add_sample(task_train,
                                     obs,
                                     action,
                                     reward,
                                     terminal,
                                     next_obs,
                                     **{'env_info': {}})

    # Init replay buffer
    Init_buffer(prm, task, replay_buffer)
    # replay_buffer.task_buffers[0].convert_D4RL(d4rl.qlearning_dataset(env))

    # Create model
    actor_model = get_actor_model(prm, prm.hidden_sizes)
    critic_model = get_critic_model(prm, prm.hidden_sizes)
    if prm.is_load_model:
        critic_model.load_state_dict(torch.load(prm.model_dir)['critic_prior_model'])

    # Create agent
    agent = TD3_BC(prm, actor_model, critic_model, replay_buffer, env, task, max_action,
                   prm.tau, prm.gamma, prm.policy_noise*max_action, prm.noise_clip*max_action,
                   prm.alpha)

    episode_rewards = []
    total_time_steps = []

    avg_reward = 0
    avg_step = 0
    for _ in range(10):
        info = agent.run_episode()
        avg_reward += info['episode_reward']
        avg_step += info['episode_timesteps']
    print("the ep_r is {}, the t is {}, the train_steps is {}".format(avg_reward / 10, int(avg_step / 10), 0))

    for t in range(prm.total_timesteps):
        agent.do_train()
        # Evaluate episode
        if (t+1) % prm.eval_freq == 0:
            avg_reward = 0
            avg_step = 0
            for _ in range(10):
                info = agent.run_episode()
                avg_reward += info['episode_reward']
                avg_step += info['episode_timesteps']
            episode_rewards.append(avg_reward / 10)
            total_time_steps.append(t+1)
            print("the ep_r is {}, the t is {}, the train_steps is {}".format(avg_reward / 10, int(avg_step / 10), t+1))

    def plot(timesteps, return_reward):
        import matplotlib.pyplot as plt

        plt.plot(timesteps, return_reward)
        plt.xlabel("alpha = " + str(prm.alpha))
        # plt.xlabel("actor_hidden_sizes = " + str(prm.hidden_sizes) + " batch_size = " + str(prm.batch_size))
        plt.show()

    plot(total_time_steps, episode_rewards)

    # 保存训练评估数据
    data = {}
    data['episode_rewards'] = episode_rewards
    data['timesteps'] = total_time_steps

    save_dir = '../../evaluate'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if prm.is_load_model:
        save_dir = os.path.join(save_dir, 'meta_critic{}.npy'.format(prm.seed))
    else:
        save_dir = os.path.join(save_dir, 'vanilla{}.npy'.format(prm.seed))
    np.save(save_dir, data)