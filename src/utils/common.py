import os
import pickle
import random
import sys
from collections import namedtuple
from datetime import datetime

import gym
import numpy as np
import torch


def grad_step(objective, optimizer, lr_schedule=None, initial_lr=None, i_epoch=None):
    if lr_schedule:
        adjust_learning_rate_schedule(optimizer, i_epoch, initial_lr, **lr_schedule)

    optimizer.zero_grad()
    objective.backward()
    optimizer.step()

def adjust_meta_factor_schedule(timestep, initial_mf, decay_factor, decay_epochs):
    # Find the index of the current interval:
    interval_index = len([mark for mark in decay_epochs if mark < timestep])       # TODO

    meta_factor = initial_mf * (decay_factor ** interval_index)

    return meta_factor

def adjust_learning_rate_schedule(optimizer, epoch, initial_lr, decay_factor, decay_epochs):
    """The learning rate is decayed by decay_factor at each interval start """

    # Find the index of the current interval:
    interval_index = len([mark for mark in decay_epochs if mark < epoch])

    lr = initial_lr * (decay_factor ** interval_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# -----------------------------------------------------------------------------------------------------------#
# sample_tasks
# -----------------------------------------------------------------------------------------------------------#

def sample_tasks(prm, env, num_tasks, train_mode=True):

    if prm.env_name == 'Maze-v0':
        if train_mode:
            data_path = './maze_data/maze_1k_' + prm.run_name.split('_')[-1] + '.npz'
        else:
            data_path = './maze_data/maze_1k_test_' + prm.run_name.split('_')[-1] + '.npz'
        tasks = np.load(data_path, allow_pickle=True)['arr_0']
        maze_idx = np.random.choice(a=len(tasks), size=num_tasks, replace=False)
        return tasks[maze_idx].tolist()
    return env.unwrapped.sample_tasks(num_tasks)

def sample_env_tasks(prm, env):
    '''
        Sample env tasks
    '''
    if prm.enable_promp_envs == True:
        # task list created as [ train_task,..., train_task ,eval_task,..., eval_task]
        train_tasks = env.sample_tasks(prm.n_train_tasks)
        eval_tasks = env.sample_tasks(prm.n_eval_tasks)
    else:
        # task list created as [ train_task,..., train_task ,eval_task,..., eval_task]
        tasks = env.get_all_task_idx()
        train_tasks = list(tasks[:prm.n_train_tasks])
        eval_tasks = list(tasks[-prm.n_eval_tasks:])

    return train_tasks, eval_tasks

# -----------------------------------------------------------------------------------------------------------#
# Prints
# -----------------------------------------------------------------------------------------------------------#

def status_string(i_epoch, num_epochs, batch_idx, n_batches, loss_data, batch_reward):

    progress_per = 100. * (i_epoch * n_batches + batch_idx) / (n_batches * num_epochs)

    return ('({:2.1f}%)\tEpoch: {} \t Batch: {} \t Objective: {:.4} \t  Reward: {:.3f}\t'.format(
        progress_per, i_epoch, batch_idx, loss_data, float(batch_reward)))

def write_to_log(message, prm, mode='a', update_file=True):
    # mode='a' is append
    # mode = 'w' is write new file
    if not isinstance(message, list):
        message = [message]
    # update log file: 更新日志文件
    if update_file:
        log_file_path = os.path.join(prm.result_dir, 'log') + '.out'
        with open(log_file_path, mode) as f:
            for string in message:
                print(string, file=f)
    # print to console:
    for string in message:
        print(string)


def write_final_result(test_avg_return, run_time, prm, result_name='', verbose=1):
    message = []
    if verbose == 1:
        message.append('Run finished at: ' + datetime.now().strftime(' %Y-%m-%d %H:%M:%S'))
    message.append(result_name + ' Average Test Reward: {:.4}\t Runtime: {:.1f} [sec]'
                     .format(test_avg_return, run_time))
    write_to_log(message, prm)

# -----------------------------------------------------------------------------------------------------------#
# Result saving
# -----------------------------------------------------------------------------------------------------------#

def setup_logAndCheckpoints(args):

    fname = str.lower(args.env_name)
    fname_log = os.path.join(args.result_dir, fname)
    fname_eval = os.path.join(fname_log, 'eval.csv')
    fname_adapt = os.path.join(fname_log, 'adapt.csv')

    return fname_log, fname_eval, fname_adapt

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def create_result_dir(prm, run_experiments=True):

    if run_experiments:
        # If run_name empty, set according to time
        time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        if prm.run_name == '':
            prm.run_name = time_str
        # prm.result_dir = os.path.join(os.curdir, 'saved', prm.run_name)       # TODO
        prm.result_dir = os.path.join(os.curdir, prm.run_name, time_str)
        if not os.path.exists(prm.result_dir):
            os.makedirs(prm.result_dir)
        message = [
                   'Run script: ' + sys.argv[0],
                   'Log file created at ' + time_str,
                   'Parameters:', str(prm), '-' * 70]
        # 打印日志信息
        write_to_log(message, prm, mode='w') # create new log file
        write_to_log('Results dir: ' + prm.result_dir, prm)
        write_to_log('-'*50, prm)

        if not hasattr(prm, 'load_model_path') or prm.load_model_path == '':
            prm.load_model_path = os.path.join(prm.result_dir, 'model.pt')
    else:
        # In this case just check if result dir exists and print the loaded parameters
        prm.result_dir = os.path.join('saved', prm.run_name)
        if not os.path.exists(prm.result_dir):
            raise ValueError('Results dir not found:  ' + prm.result_dir)
        else:
            print('Run script: ' + sys.argv[0])
            print( 'Data loaded from: ' + prm.result_dir)
            print('-' * 70)

def save_model_state(model, f_path):
    with open(f_path, 'wb') as f_pointer:
        torch.save(model.state_dict(), f_pointer)
    return f_path

def save_models_state(models, f_path):
    with open(f_path, 'wb') as f_pointer:
        torch.save(models, f_pointer)
    return f_path

def load_model_state(model, f_path):
    if not os.path.exists(f_path):
        raise ValueError('No file found with the path: ' + f_path)
    with open(f_path, 'rb') as f_pointer:
        model.load_state_dict(torch.load(f_pointer))

def save_run_data(prm, info_dict):
    run_data_file_path = os.path.join(prm.result_dir, 'run_data.pkl')
    with open(run_data_file_path, 'wb') as f:
        pickle.dump([prm, info_dict], f)

def create_env(prm):

    from src.misc.env_meta import build_PEARL_envs
    env = build_PEARL_envs(seed=prm.seed,
                           env_name=prm.env_name,
                           params=prm)

    return env



