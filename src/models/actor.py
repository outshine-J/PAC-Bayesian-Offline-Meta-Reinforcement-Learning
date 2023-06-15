
import torch
import torch.nn.functional as F

from src.models.policy import Policy, StochasticLinear

class Actor(Policy):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 hidden_sizes=(64, 64),
                 log_var_init=None,
                 nonlinearity=F.relu
                 ):
        super(Actor, self).__init__(input_size=state_dim, output_size=action_dim)

        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)+1
        self.nonlinearity = nonlinearity

        self.max_action = max_action

        layer_sizes = (state_dim,) + hidden_sizes
        for i in range(1, self.num_layers):
            self.add_module('layer{0}'.format(i),
                            StochasticLinear(layer_sizes[i - 1], layer_sizes[i], log_var_init=log_var_init))

        self.out = StochasticLinear(layer_sizes[-1], action_dim, log_var_init=log_var_init)

    def forward(self, state):

        output = state

        for i in range(1, self.num_layers):
            output = getattr(self, 'layer{0}'.format(i))(output)
            output = self.nonlinearity(output)

        action = self.max_action * torch.tanh(self.out(output))

        return action


def get_actor_model(prm, hidden_size=(64, 64)):
    env = prm.env
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    model = Actor(state_dim=state_dim,
                  action_dim=action_dim,
                  max_action=max_action,
                  hidden_sizes=hidden_size,
                  log_var_init=prm.log_var_init).to(prm.device)

    return model