import argparse
import os

import torch
from torch import optim

from src.alogs import meta_rl
from src.utils.common import create_result_dir, create_env, sample_env_tasks, set_random_seed, save_models_state
from src.utils.util import config_tasks_envs

parser = argparse.ArgumentParser()

# Run Parameters
parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)', default='Ant-dir')
parser.add_argument('--seed', type=int, help='random seed', default=10)
parser.add_argument('--gpu_index', type=int, help='The index of GPU device to run on', default=0)
parser.add_argument('--env_configs', type=str, help='config file path', default='./configs/ant-dir.json')
parser.add_argument('--data_dir', type=str, help='Task replay buffer and Information Paths', default=None)
parser.add_argument('--train_epoch', type=int, help='corresponding epoch of the model used to generate meta-training trajectories', default=6e5)
parser.add_argument('--eval_epoch', type=int, help='corresponding epoch of the model used to generate meta-testing trajectories', default=6e5)
parser.add_argument('--sample', type=int, help='whether to train with stochastic (noise-sampled) trajectories, for offline method', default=1)
parser.add_argument('--n_trj', type=int, default=50)

# Env Parameters
parser.add_argument('--env_name', type=str, help='Ant-dir or Cheetah-vel et al.', default="ant-dir")
parser.add_argument('--max_path_length', type=int, help='Maximum path length per episode', default=200)
parser.add_argument('--log_interval', type=int, default=1, help='log interval, one log per n updates')
parser.add_argument('--goal_radius', type=float, default=1.0)

# Task Parameters
parser.add_argument('--n_train_tasks', type=int, help="Number of tasks used for training", default=10)
parser.add_argument('--n_eval_tasks', type=int, help="Number of tasks used for evaluation", default=10)
parser.add_argument('--n_tasks', type=int, help="Total number of environmental tasks", default=20)
parser.add_argument('--num_evals', type=int, help="the number of evaluate tasks", default=10)
parser.add_argument('--offline_evaluate', type=bool, help='whether to take offline evaluation', default=True)
parser.add_argument('--enable_promp_envs', type=bool, help='whether to randomly generate a batch of tasks', default=False)

# Algo Parameters
parser.add_argument('--total_timesteps', type=int, help="total number of timesteps to train on", default=0.4e6)
parser.add_argument('--num_train_steps_per_itr', type=int, help='the number of meta_gradient steps taken per iteration', default=200)
parser.add_argument('--eval_freq', type=int, help='How often (time steps) we evaluate', default=1e3)
parser.add_argument('--lr', type=float, help='initial learning rate', default=3e-4)
parser.add_argument('--replay_size', default=int(1e4), type=int, help='Replay buffer size int(2000)')
parser.add_argument('--batch_size', type=int, help="the number of samples used to calculate the loss", default=256)
parser.add_argument('--hidden_sizes', type=tuple, help="The sizes of model's hidden_layers", default=(300, 300, 300))
parser.add_argument('--enable_train_eval', type=bool, help='whether to evaluate the training task', default=False)
parser.add_argument('--enable_adaptation', type=bool, help='whether to make adaptive update in the evaluation stage', default=True)
parser.add_argument('--snap_iter_nums', type=int, help="how many times adapt using eval task", default=10)
parser.add_argument('--snap_init_steps', type=int, default=600)

# TD3-BC Algo Parameters
parser.add_argument('--gamma', type=float, help='discount factor', default=0.99)
parser.add_argument('--tau', type=float, help='soft policy update', default=0.005)
parser.add_argument('--policy_noise', type=float, help='Noise added to target policy', default=0.1)
parser.add_argument('--noise_clip', type=float, help='Range to clip target policy noise', default=0.5)
parser.add_argument('--policy_freq', type=int, help='Frequency of delayed policy updates', default=2)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--normalize', type=bool, default=True)
parser.add_argument('--decay_step', type=int, help='Learning rate attenuation step size', default=50000)

prm = parser.parse_args()
prm.device = torch.device("cuda:" + str(prm.gpu_index) if torch.cuda.is_available() else "cpu")
prm.log_var_init = {'mean': -10.0, 'std': 0.1}

prm.wc = 1.0
prm.bound = 1.0

# read tasks/env config params and update args
config_tasks_envs(prm)

prm.n_MC = 1
prm.optim_func, prm.optim_args = optim.Adam, {'lr': prm.lr}
prm.lr_schedule = {}
# parameter of the hyper-prior regularization
prm.kappa_prior = 2e3
prm.kappa_post = 1e-3
prm.delta = 0.1

def main():
    ###############
    #  Init run   #
    ###############
    create_result_dir(prm)

    # path to save the learned meta-parameters
    save_path = os.path.join(prm.result_dir, prm.env_name, 'model.pt')

    # Create the environment
    env = create_env(prm)
    env.close()
    prm.env = env

    # Set random seed
    set_random_seed(prm.seed)

    # Sampling a batch of meta-learning tasks
    meta_train_tasks, meta_eval_tasks = sample_env_tasks(prm, env=prm.env)

    # Create a meta-learning framework
    alg = meta_rl.MetaLearner(prm, meta_train_tasks=meta_train_tasks,
                              meta_eval_tasks=meta_eval_tasks,
                              hidden_sizes=prm.hidden_sizes,
                              total_timesteps=prm.total_timesteps)

    # Run Meta-Training
    prior_model = alg.meta_train()

    # save model
    save_models_state(prior_model, save_path)

if __name__ == "__main__":
    main()






