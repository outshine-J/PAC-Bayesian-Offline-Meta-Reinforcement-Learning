import numpy as np
import torch


class SimpleReplayBuffer():
    def __init__(self, max_replay_buffer_size, observation_dim, action_dim, goal_radius=1.0, device='cpu'):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._sparse_rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        self.goal_radius = goal_radius
        self.device = device
        self.clear()

    def add_path(self, path):
        """
        Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by rlkit.samplers.util.rollout
        """
        for i, (
                obs,
                action,
                reward,
                next_obs,
                terminal,
                # agent_info,
                # env_info
        ) in enumerate(zip(
            path["observations"],
            path["actions"],
            path["rewards"],
            path["next_observations"],
            path["terminals"],
            # path["agent_infos"],
            # path["env_infos"],
        )):
            self.add_sample(
                obs,
                action,
                reward,
                terminal,
                next_obs,
                # agent_info=agent_info,
                # env_info=env_info,
                **{'env_info': {}}
            )
        self.terminate_episode()

    def add_trajectory(self, trajectory):
        '''
            trajectory: [states, actions, rewards, next_rewards, dones]
        '''
        for idx, experience in enumerate(trajectory[:]):
            state, action, reward, next_state, done = experience
            self.add_sample(state, action, reward, done, next_state, **{'env_info': {}})
        self.terminate_episode()

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        if reward >= self.goal_radius:
            sparse_reward = (reward - self.goal_radius) * (1/abs(self.goal_radius))
            self._sparse_rewards[self._top] = kwargs['env_info'].get('sparse_reward', sparse_reward)
        else:
            self._sparse_rewards[self._top] = kwargs['env_info'].get('sparse_reward', 0)
        self._advance()

    def terminate_episode(self):
        # store the episode beginning once the episode is over
        # n.b. allows last episode to loop but whatever
        self._episode_starts.append(self._cur_episode_start)
        self._cur_episode_start = self._top

    def size(self):
        return self._size

    def num_steps_can_sample(self):
        return self._size

    def clear(self):
        self._top = 0
        self._size = 0
        self._episode_starts = []
        self._cur_episode_start = 0

    def empty_buffer(self):
        self._observations = np.zeros_like(self._observations)
        self._next_obs = np.zeros_like(self._next_obs)
        self._actions = np.zeros_like(self._actions)
        self._rewards = np.zeros_like(self._rewards)
        self._sparse_rewards = np.zeros_like(self._sparse_rewards)
        self._terminals = np.zeros_like(self._terminals)
        self.clear()

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def sample_data(self, indices):

        states = self._observations[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_obs[indices]
        terminals = self._terminals[indices]

        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(actions, dtype=np.float32)).to(self.device)
        rewards = torch.from_numpy(np.array(rewards, dtype=np.float32)).reshape(-1, 1).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        terminals = torch.from_numpy(np.array(terminals, dtype=np.float32)).reshape(-1, 1).to(self.device)

        return states, actions, rewards, next_states, terminals

    def random_batch(self, batch_size):
        ''' batch of unordered transitions '''
        assert self._size > 0
        indices = np.random.randint(0, self._size, batch_size)
        return self.sample_data(indices)

    def random_sequence(self, batch_size):
        ''' batch of trajectories '''
        # take random trajectories until we have enough
        i = 0
        indices = []
        while len(indices) < batch_size:
            start = np.random.choice(self.episode_starts[:-1])
            pos_idx = self._episode_starts.index(start)
            indices += list(range(start, self._episode_starts[pos_idx + 1]))
            i += 1
        # cut off the last traj if needed to respect batch size
        indices = indices[:batch_size]
        return self.sample_data(indices)

    def normalize_states(self, eps=1e-3):
        mean = self._observations.mean(0, keepdims=True)
        std = self._observations.std(0, keepdims=True) + eps
        self._observations = (self._observations - mean) / std
        self._next_obs = (self._next_obs - mean) / std
        return mean, std

    def convert_D4RL(self, dataset):
        for i in range(dataset['observations'].shape[0]):
            state = dataset['observations'][i]
            action = dataset['actions'][i]
            next_state = dataset['next_observations'][i]
            reward = dataset['rewards'][i].reshape(-1, 1)
            mark = dataset['terminals'][i].reshape(-1, 1)
            self.add_sample(state, action, reward, mark, next_state, **{'env_info': {}})