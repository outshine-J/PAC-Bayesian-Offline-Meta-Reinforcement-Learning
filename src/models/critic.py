
from src.models.policy import Policy, StochasticLinear

import torch
import torch.nn.functional as F

class Critic(Policy):
    """
    Simple critic network.
    Will create an critic operated in continuous action space.
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_sizes=(64, 64),
                 log_var_init=None,
                 nonlinearity=F.relu):
        super(Critic, self).__init__(input_size=state_dim + action_dim, output_size=1)

        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)+1
        self.nonlinearity = nonlinearity

        layer_sizes = (state_dim + action_dim,) + hidden_sizes
        # Q1 architecture
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            StochasticLinear(layer_sizes[i - 1], layer_sizes[i], log_var_init=log_var_init))
        self.q1_out = StochasticLinear(layer_sizes[-1], 1, log_var_init=log_var_init)

        # Q2 architecture
        j = self.num_layers
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(j+i-1),
                            StochasticLinear(layer_sizes[i - 1], layer_sizes[i], log_var_init=log_var_init))
        self.q2_out = StochasticLinear(layer_sizes[-1], 1, log_var_init=log_var_init)

    def forward(self, state, action):
        s_a = torch.cat([state, action], 1)

        Q1_output = s_a
        for i in range(1, self.num_layers):
            Q1_output = getattr(self, 'layer{0}'.format(i))(Q1_output)
            Q1_output = self.nonlinearity(Q1_output)
        Q1 = self.q1_out(Q1_output)

        j = self.num_layers
        Q2_output = s_a
        for i in range(1, self.num_layers):
            Q2_output = getattr(self, 'layer{0}'.format(j+i-1))(Q2_output)
            Q2_output = self.nonlinearity(Q2_output)
        Q2 = self.q2_out(Q2_output)

        return Q1, Q2

    def Q1(self, state, action):
        s_a = torch.cat([state, action], 1)

        Q1_output = s_a
        for i in range(1, self.num_layers):
            Q1_output = getattr(self, 'layer{0}'.format(i))(Q1_output)
            Q1_output = self.nonlinearity(Q1_output)
        Q1 = self.q1_out(Q1_output)

        return Q1

def get_critic_model(prm, hidden_size=(64, 64)):
    env = prm.env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = Critic(state_dim=state_dim,
                   action_dim=action_dim,
                   hidden_sizes=hidden_size,
                   log_var_init=prm.log_var_init).to(prm.device)

    return model