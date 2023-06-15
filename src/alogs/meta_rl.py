import glob
import os
import time
from copy import deepcopy

import numpy as np
import torch
from matplotlib.ticker import MultipleLocator
import torch.optim.lr_scheduler as lr_scheduler

from src.alogs.batch_agent import MultiTaskAgent
from src.alogs.get_object_MPB import get_hyper_divergnce, get_task_complexity, get_meta_complexity_term
from src.models.actor import get_actor_model
from src.models.critic import get_critic_model
from src.rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from src.utils import logger
from src.utils.common import setup_logAndCheckpoints, grad_step, adjust_meta_factor_schedule
from src.utils.util import CSVWriter

class MetaLearner(object):
    def __init__(self,
                 prm,
                 meta_train_tasks,
                 meta_eval_tasks,
                 hidden_sizes=(64, 64),
                 total_timesteps=5e6
                 ):

        self.prm = prm
        self.env = prm.env
        self.device = prm.device
        self.data_dir = prm.data_dir

        # setup tasks data
        self.meta_train_tasks = meta_train_tasks
        self.meta_eval_tasks = meta_eval_tasks

        # setup some params
        self.n_trj = prm.n_trj
        self.train_epoch = prm.train_epoch
        self.eval_epoch = prm.eval_epoch
        self.sample = prm.sample
        self.max_path_length = prm.max_path_length
        self.offline_evaluate = prm.offline_evaluate
        self.total_timesteps = total_timesteps
        self.num_train_steps_per_itr = prm.num_train_steps_per_itr
        self.eval_freq = prm.eval_freq
        self.num_evals = prm.num_evals
        self.snap_iter_nums = prm.snap_iter_nums
        self.snap_init_steps = prm.snap_init_steps
        self.policy_freq = prm.policy_freq

        # Create prior model
        self.actor_prior_model = get_actor_model(prm, hidden_sizes)
        self.critic_prior_model = get_critic_model(prm, hidden_sizes)

        # Create batch_post train model
        self.actor_posteriors_train_models = [get_actor_model(prm, hidden_sizes) for _ in range(len(meta_train_tasks))]
        self.critic_posteriors_train_models = [get_critic_model(prm, hidden_sizes) for _ in range(len(meta_train_tasks))]

        # Create batch_post eval model
        self.actor_posteriors_eval_models = [get_actor_model(prm, hidden_sizes) for _ in range(len(meta_eval_tasks))]
        self.critic_posteriors_eval_models = [get_critic_model(prm, hidden_sizes) for _ in range(len(meta_eval_tasks))]

        # Create replay_buffer
        self.train_buffer = MultiTaskReplayBuffer(self.prm.replay_size, self.env, self.meta_train_tasks, self.prm.goal_radius, device=self.device)
        self.eval_buffer = MultiTaskReplayBuffer(self.prm.replay_size, self.env, self.meta_eval_tasks, self.prm.goal_radius, device=self.device)
        self.init_buffer()

        # Create batch_post agent
        #
        self.batch_train_agent = MultiTaskAgent(prm,
                                                env=self.env,
                                                batch_actor_model=self.actor_posteriors_train_models,
                                                batch_critic_model=self.critic_posteriors_train_models,
                                                batch_tasks=self.meta_train_tasks,
                                                multi_replay_buffer=self.train_buffer,
                                                )

        self.batch_eval_agent = MultiTaskAgent(prm,
                                               env=self.env,
                                               batch_actor_model=self.actor_posteriors_eval_models,
                                               batch_critic_model=self.critic_posteriors_eval_models,
                                               batch_tasks=self.meta_eval_tasks,
                                               multi_replay_buffer=self.eval_buffer
                                               )

        optim_func, optim_args, lr_schedule = \
            prm.optim_func, prm.optim_args, prm.lr_schedule

        # Create model optimizer
        all_actor_train_post_params = sum([list(posterior_model.parameters()) for posterior_model in self.actor_posteriors_train_models], [])
        actor_prior_params = list(self.actor_prior_model.parameters())
        all_actor_params = all_actor_train_post_params + actor_prior_params
        self.all_actor_optimizer = optim_func(all_actor_params, **optim_args)
        # self.actor_lr_schedule = lr_scheduler.CosineAnnealingLR(self.all_actor_optimizer, T_max=int(self.total_timesteps / self.policy_freq), eta_min=1e-6)
        self.actor_lr_schedule = lr_scheduler.StepLR(self.all_actor_optimizer, step_size=int(prm.decay_step / self.policy_freq), gamma=0.1)

        all_critic_train_post_params = sum([list(posterior_model.parameters()) for posterior_model in self.critic_posteriors_train_models], [])
        critic_prior_params = list(self.critic_prior_model.parameters())
        all_critic_params = all_critic_train_post_params + critic_prior_params
        self.all_critic_optimizer = optim_func(all_critic_params, **optim_args)
        # self.critic_lr_schedule = lr_scheduler.CosineAnnealingLR(self.all_critic_optimizer, T_max=int(self.total_timesteps), eta_min=1e-6)
        self.critic_lr_schedule = lr_scheduler.StepLR(self.all_critic_optimizer, step_size=int(prm.decay_step), gamma=0.1)

        self.meta_schedule = prm.meta_schedule
        self.meta_factor = 1.0
        self.initial_mf = 1.0

    def init_buffer(self):

        train_trj_paths = []
        eval_trj_paths = []
        # trj entry format: [obs, action, reward, new_obs]
        if hasattr(self, 'sample') and self.sample:
            for n in range(self.n_trj):
                if hasattr(self, 'train_epoch') and self.train_epoch:
                    train_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step%d.npy" % (n, self.train_epoch)))
                else:
                    train_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" % (n)))
                if hasattr(self, 'eval_epoch') and self.eval_epoch:
                    eval_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step%d.npy" %(n, self.eval_epoch)))
                else:
                    eval_trj_paths += glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_evalsample%d_step*.npy" %(n)))
        else:
            if hasattr(self, 'train_epoch') and self.train_epoch:
                train_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step%d.npy" %(self.n_trj, self.train_epoch)))
            else:
                train_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step*.npy") %(self.n_trj))
            if hasattr(self, 'eval_epoch') and self.eval_epoch:
                eval_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step%d.npy" %(self.n_trj, self.eval_epoch)))
            else:
                eval_trj_paths = glob.glob(os.path.join(self.data_dir, "goal_idx*", "trj_eval[0-%d]_step*.npy") %(self.n_trj))

        train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.meta_train_tasks]
        train_task_idxs = [int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) for train_trj_path in train_trj_paths if
                           int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.meta_train_tasks]
        eval_paths = [eval_trj_path for eval_trj_path in eval_trj_paths if
                      int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.meta_eval_tasks]
        eval_task_idxs = [int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) for eval_trj_path in eval_trj_paths if
                          int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.meta_eval_tasks]

        # training task list
        obs_train_lst = []
        action_train_lst = []
        reward_train_lst = []
        next_obs_train_lst = []
        terminal_train_lst = []
        task_train_lst = []
        # eval task list
        obs_eval_lst = []
        action_eval_lst = []
        reward_eval_lst = []
        next_obs_eval_lst = []
        terminal_eval_lst = []
        task_eval_lst = []

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
        for eval_path, eval_task_idx in zip(eval_paths, eval_task_idxs):
            trj_npy = np.load(eval_path, allow_pickle=True)
            obs_eval_lst += list(trj_npy[:, 0])
            action_eval_lst += list(trj_npy[:, 1])
            reward_eval_lst += list(trj_npy[:, 2])
            next_obs_eval_lst += list(trj_npy[:, 3])
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_eval_lst += terminal
            task_eval = [eval_task_idx for _ in range(trj_npy.shape[0])]
            task_eval_lst += task_eval

        # load training buffer
        for i, (task_train, obs, action, reward, next_obs, terminal) in \
                enumerate(zip(task_train_lst,
                              obs_train_lst,
                              action_train_lst,
                              reward_train_lst,
                              next_obs_train_lst,
                              terminal_train_lst)):
            self.train_buffer.add_sample(task_train,
                                         obs,
                                         action,
                                         reward,
                                         terminal,
                                         next_obs,
                                         **{'env_info': {}})
        # load evaluation buffer
        for i, (task_eval, obs, action, reward, next_obs, terminal) in \
                enumerate(zip(task_eval_lst,
                              obs_eval_lst,
                              action_eval_lst,
                              reward_eval_lst,
                              next_obs_eval_lst,
                              terminal_eval_lst)):
            self.eval_buffer.add_sample(task_eval,
                                        obs,
                                        action,
                                        reward,
                                        terminal,
                                        next_obs,
                                        **{'env_info': {}})

    def meta_train(self):
        """
            Meta_training loop
        """
        ### Set log folder
        log_file_dir, fname_csv_eval, fname_adapt = setup_logAndCheckpoints(self.prm)
        logger.configure(dir=log_file_dir)
        ##################################
        # Train and eval
        ##################################

        # Step 1: Define the variables that need to be logged
        # define some req vars
        timesteps_since_eval = 0
        update_iter = 0
        sampling_loop = 0
        all_timesteps = [update_iter]

        # Step 2: Eval_tasks and train_tasks are evaluated before training starts
        # Evaluate untrained policy
        eval_data_mean = self.evaluate_policy(eval_tasks=self.meta_eval_tasks, mode='eval')
        eval_results_mean = [eval_data_mean]

        if self.prm.enable_train_eval:
            train_subset = np.random.choice(a=self.meta_train_tasks, size=len(self.meta_eval_tasks), replace=False)
            train_subset_tasks_eval = self.evaluate_policy(eval_tasks=train_subset, mode='train')
            train_results_mean = [train_subset_tasks_eval]
        else:
            train_subset_tasks_eval = 0
            train_results_mean = [train_subset_tasks_eval]

        wrt_csv_eval = CSVWriter(fname_csv_eval, {'nupdates': update_iter,
                                                  'total_timesteps': update_iter,
                                                  'eval_eprewmean': eval_results_mean[0],
                                                  'train_eprewmean': train_subset_tasks_eval,
                                                  'sampling_loop': sampling_loop
                                                  })

        wrt_csv_eval.write({'nupdates': update_iter,
                            'total_timesteps': update_iter,
                            'eval_eprewmean': eval_results_mean[0],
                            'train_eprewmean': train_subset_tasks_eval,
                            'sampling_loop': sampling_loop
                            })

        # Start total timer
        tstart = time.time()

        # Step 3: training and evaluation loop
        ####
        # Train and eval main loop
        ####
        while update_iter < self.prm.total_timesteps:

            #######
            # run training to calculate loss, run backward, and update params
            #######
            sampling_loop += 1

            # train
            alg_stats = self.train(iterations=self.num_train_steps_per_itr)
            update_iter += alg_stats['timesteps']
            timesteps_since_eval += alg_stats['timesteps']

            self.meta_factor = adjust_meta_factor_schedule(update_iter, self.initial_mf, self.meta_schedule['decay_factor'], self.meta_schedule['decay_epochs'])


            if sampling_loop % 1 == 0:
                logger.record_tabular("nupdates", update_iter)
                logger.record_tabular("total_timesteps", update_iter)
                logger.record_tabular("actor_empiric_loss", float(alg_stats['actor_avg_empiric_loss']))
                logger.record_tabular("actor_task_complexity", float(alg_stats['actor_task_complexity']))
                logger.record_tabular("actor_meta_complexity", float(alg_stats['actor_meta_complexity']))
                logger.record_tabular("critic_empiric_loss", float(alg_stats['critic_avg_empiric_loss']))
                logger.record_tabular("critic_task_complexity", float(alg_stats['critic_task_complexity']))
                logger.record_tabular("critic_meta_complexity", float(alg_stats['critic_meta_complexity']))
                logger.record_tabular("sampling_loop", sampling_loop)

                logger.dump_tabular()

            #######
            # run eval
            #######
            if timesteps_since_eval >= self.eval_freq:
                all_timesteps.append(update_iter)
                timesteps_since_eval %= self.eval_freq
                eval_temp = self.evaluate_policy(eval_tasks=self.meta_eval_tasks, mode='eval')
                eval_results_mean.append(eval_temp)

                # Eval subset of train tasks
                if self.prm.enable_train_eval:
                    train_subset = np.random.choice(a=self.meta_train_tasks, size=len(self.meta_eval_tasks), replace=False)
                    train_subset_tasks_eval = self.evaluate_policy(eval_tasks=train_subset, mode='train')
                    train_results_mean.append(train_subset_tasks_eval)

                else:
                    train_subset_tasks_eval = 0.0
                    train_results_mean.append(train_subset_tasks_eval)
                # dump results
                wrt_csv_eval.write({'nupdates': update_iter,
                                    'total_timesteps': update_iter,
                                    'eval_eprewmean': eval_temp,
                                    'train_eprewmean': train_subset_tasks_eval,
                                    'sampling_loop': sampling_loop
                                    })

        # end while
        all_timesteps.append(update_iter)
        # Step 4: Re-evaluate after training
        ###############
        # Eval for the final time
        ###############
        eval_temp = self.evaluate_policy(eval_tasks=self.meta_eval_tasks, mode='eval')

        # Eval subset of train tasks:
        if self.prm.enable_train_eval:
            train_subset = np.random.choice(a=self.meta_train_tasks, size=len(self.meta_eval_tasks), replace=False)
            train_subset_tasks_eval = self.evaluate_policy(eval_tasks=train_subset, mode='train')
            train_results_mean.append(train_subset_tasks_eval)
        else:
            train_subset_tasks_eval = 0.0
            train_results_mean.append(train_subset_tasks_eval)

        eval_results_mean.append(eval_temp)

        wrt_csv_eval.write({'nupdates': update_iter,
                            'total_timesteps': update_iter,
                            'eval_eprewmean': eval_temp,
                            'train_eprewmean': train_subset_tasks_eval,
                            'sampling_loop': sampling_loop
                            })
        wrt_csv_eval.close()
        print("train finish")
        self.plot(time_steps=all_timesteps, eval_returns=eval_results_mean, mode="Eval")
        self.plot(time_steps=all_timesteps, eval_returns=train_results_mean, mode="Train")

        return {"actor_prior_model": self.actor_prior_model.state_dict(),
                "critic_prior_model": self.critic_prior_model.state_dict()}

    def train(self, iterations):

        for it in range(1, iterations+1):

            n_tasks_in_mb = len(self.meta_train_tasks)

            actor_hyper_dvrg = get_hyper_divergnce(self.prm, self.actor_prior_model)
            critic_hyper_dvrg = get_hyper_divergnce(self.prm, self.critic_prior_model)

            actor_avg_empiric_loss_per_task = torch.zeros(len(self.meta_train_tasks), device=self.prm.device)
            critic_avg_empiric_loss_per_task = torch.zeros(len(self.meta_train_tasks), device=self.prm.device)

            actor_complexity_per_task = torch.zeros(len(self.meta_train_tasks), device=self.prm.device)
            critic_complexity_per_task = torch.zeros(len(self.meta_train_tasks), device=self.prm.device)

            n_samples_per_task = torch.zeros(len(self.meta_train_tasks), device=self.prm.device)

            for i_task, task_idx in enumerate(self.meta_train_tasks):
                # calculate average_empiric_loss
                actor_avg_empiric_loss, critic_avg_empiric_loss = 0.0, 0.0
                for i_MC in range(self.prm.n_MC):
                    batch_samples = self.batch_train_agent.batch_agent[task_idx].batch_sample()
                    critic_loss = self.batch_train_agent.batch_agent[task_idx].calculate_critic_losses(batch_samples)
                    critic_avg_empiric_loss += critic_loss
                    if it % self.policy_freq == 0:
                        actor_loss = self.batch_train_agent.batch_agent[task_idx].calculate_actor_losses(batch_samples)
                        actor_avg_empiric_loss += actor_loss
                actor_avg_empiric_loss /= self.prm.n_MC
                critic_avg_empiric_loss /= self.prm.n_MC

                actor_avg_empiric_loss_per_task[i_task] = actor_avg_empiric_loss
                critic_avg_empiric_loss_per_task[i_task] = critic_avg_empiric_loss

                n_samples = self.batch_train_agent.batch_agent[task_idx].replay_buffer.size()
                n_samples_per_task[task_idx] = n_samples

                # calculate task_complex_term
                model_idx = self.batch_train_agent.corresponding_model[task_idx]
                critic_complexity = get_task_complexity(self.prm, prior_model=self.critic_prior_model,
                                                                 post_model=self.critic_posteriors_train_models[model_idx],
                                                                 n_samples=n_samples,
                                                                 n_tasks=n_tasks_in_mb,
                                                                 avg_empiric_loss=critic_avg_empiric_loss,
                                                                 hyper_dvrg=critic_hyper_dvrg, noised_prior=True)
                if it % self.policy_freq == 0:
                    actor_complexity = get_task_complexity(self.prm, prior_model=self.actor_prior_model,
                                                           post_model=self.actor_posteriors_train_models[model_idx],
                                                           n_samples=n_samples,
                                                           n_tasks=n_tasks_in_mb,
                                                           avg_empiric_loss=actor_avg_empiric_loss,
                                                           hyper_dvrg=actor_hyper_dvrg, noised_prior=True)
                    actor_complexity_per_task[i_task] = actor_complexity
                critic_complexity_per_task[i_task] = critic_complexity
            # END FOR

            # Compute meta_complex_term
            critic_meta_complex_term = get_meta_complexity_term(self.prm,
                                                                 hyper_kl=critic_hyper_dvrg,
                                                                 n_samples=n_samples_per_task.mean(),
                                                                 n_train_tasks=n_tasks_in_mb)
            if it % self.policy_freq == 0:
                actor_meta_complex_term = get_meta_complexity_term(self.prm,
                                                                   hyper_kl=actor_hyper_dvrg,
                                                                   n_samples=n_samples_per_task.mean(),
                                                                   n_train_tasks=n_tasks_in_mb)
                actor_total_objective = actor_avg_empiric_loss_per_task.mean() + actor_complexity_per_task.mean() + self.meta_factor * actor_meta_complex_term
                grad_step(actor_total_objective, self.all_actor_optimizer, initial_lr=self.prm.lr)
                self.actor_lr_schedule.step()
                # soft-update opt
                self.batch_train_agent.syn_weight()

            critic_total_objective = critic_avg_empiric_loss_per_task.mean() + critic_complexity_per_task.mean() + self.meta_factor * critic_meta_complex_term
            grad_step(critic_total_objective, self.all_critic_optimizer, initial_lr=self.prm.lr)
            self.critic_lr_schedule.step()

        # END FOR
        log = {}

        log['timesteps'] = iterations

        log['actor_avg_empiric_loss'] = actor_avg_empiric_loss_per_task.mean()
        log['actor_task_complexity'] = actor_complexity_per_task.mean()
        log['actor_meta_complexity'] = self.meta_factor * actor_meta_complex_term

        log['critic_avg_empiric_loss'] = critic_avg_empiric_loss_per_task.mean()
        log['critic_task_complexity'] = critic_complexity_per_task.mean()
        log['critic_meta_complexity'] = self.meta_factor * critic_meta_complex_term

        return log

    def plot(self, time_steps, eval_returns, mode='Eval'):
        import matplotlib.pyplot as plt

        plt.style.use('ggplot')
        figure, axs = plt.subplots(nrows=1, ncols=1)

        axs.plot(time_steps, eval_returns, linewidth=1.5, color="red")

        axs.xaxis.set_major_locator(MultipleLocator(100000))
        axs.yaxis.set_major_locator(MultipleLocator(25))

        axs.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        axs.set_xlabel("2 Meta-training Time-steps")
        axs.set_ylabel("Average %s Return" % (mode))
        axs.set_title(str(self.prm.env_name))
        axs.legend(loc='lower right')

        plt.show()

    def evaluate_policy(self, eval_tasks, mode='eval', msg='Evaluation'):
        if self.offline_evaluate:
            return self.offline_evaluate_policy(eval_tasks, mode, msg)
        else:
            return self.online_evaluate_policy(eval_tasks, mode, msg)

    def online_evaluate_policy(self, eval_tasks, mode='eval', msg='Evaluation'):
        ############# adaptation step #############
        if self.prm.enable_adaptation == True:
            self.save_model_states(mode=mode)
        ########### ############### ###############
        all_task_rewards = []
        dc_rewards = []
        if mode == 'train':
            for i, tidx in enumerate(eval_tasks):
                if self.prm.enable_adaptation:
                    avg_dc_reward = 0
                    for _ in range(self.num_evals):
                        _, traj_reward = self.rollout_policy(self.batch_train_agent.batch_agent[tidx])
                        avg_dc_reward += traj_reward
                    dc_rewards.append((avg_dc_reward / self.num_evals))
                # Ignoring the adaptive updating process of training tasks
                avg_reward = 0
                for _ in range(self.num_evals):
                    _, traj_reward = self.rollout_policy(self.batch_train_agent.batch_agent[tidx])
                    avg_reward += traj_reward
                all_task_rewards.append((avg_reward / self.num_evals))

        elif mode == 'eval':
            # Step 1: Reset the model
            for i in range(len(eval_tasks)):
                self.actor_posteriors_eval_models[i].load_state_dict(self.actor_prior_model.state_dict())
                self.critic_posteriors_eval_models[i].load_state_dict(self.critic_prior_model.state_dict())
                self.batch_eval_agent.batch_agent[i].reset_model()
            for i, tidx in enumerate(eval_tasks):
                ############# adaptation step #############
                if self.prm.enable_adaptation == True:
                    avg_dc_reward = 0
                    for _ in range(self.num_evals):
                        _, traj_reward = self.rollout_policy(self.batch_eval_agent.batch_agent[tidx])
                        avg_dc_reward += traj_reward
                    dc_rewards.append((avg_dc_reward / self.num_evals))

                    # Collect trajectory for adaptation.
                    self.collect_data_for_adaptation(self.batch_eval_agent.batch_agent[tidx])
                    model_idx = self.batch_eval_agent.corresponding_model[tidx]
                    for it in range(self.snap_iter_nums):
                        update_policy = (it % self.policy_freq == 0)
                        n_samples = self.batch_eval_agent.batch_agent[tidx].replay_buffer.size()
                        actor_complex_term = get_task_complexity(self.prm, self.actor_prior_model, self.actor_posteriors_eval_models[model_idx], n_samples)
                        critic_complex_term = get_task_complexity(self.prm, self.critic_prior_model, self.critic_posteriors_eval_models[model_idx], n_samples)
                        self.batch_eval_agent.batch_agent[tidx].adaptation(actor_complex_term, critic_complex_term, update_policy)
                avg_reward = 0
                for _ in range(self.num_evals):
                    _, traj_reward = self.rollout_policy(self.batch_eval_agent.batch_agent[tidx])
                    avg_reward += traj_reward
                all_task_rewards.append(avg_reward / self.num_evals)
        else:
            raise ValueError

        ############# adaptation step #############
        # Roll-back
        ############# ############### #############
        if self.prm.enable_adaptation == True:
            self.rollback(eval_tasks, mode)

        if self.prm.enable_adaptation == True:
            msg += ' *** with Adapation *** '
            print('Avg rewards (only one eval loop) for all tasks before adaptation ',
                  np.mean(dc_rewards))

        print("---------------------------------------")
        print("%s over %d episodes of %d %s tasks :%f" \
              % (msg, self.prm.num_evals, len(self.meta_eval_tasks), mode, np.mean(all_task_rewards)))
        print("---------------------------------------")

        return np.mean(all_task_rewards)

    def offline_evaluate_policy(self, eval_tasks, mode='eval', msg='Evaluation'):
        """
            Runs policy for X episodes and returns average reward
            eval_tasks : list
        """
        ############# adaptation step #############
        if self.prm.enable_adaptation == True:
            self.save_model_states(mode=mode)
        ########### ############### ###############

        all_task_rewards = []
        dc_rewards = []

        def evaluate(eval_agent):

            eval_agent.reset_env()

            avg_reward = 0
            for _ in range(self.prm.num_evals):

                state = eval_agent.env.reset()
                done = False
                step = 0

                while not done and step < self.prm.max_path_length:
                    action = eval_agent.select_action(state)
                    next_state, reward, done, _ = eval_agent.env.step(action)
                    state = next_state

                    reward = np.array(reward, dtype=np.float32)
                    avg_reward += reward
                    step += 1

            avg_reward = np.array(avg_reward, 'float32')
            avg_reward /= self.prm.num_evals

            return avg_reward

        if mode == 'train':
            for i, tidx in enumerate(eval_tasks):
                ############# adaptation step #############
                if self.prm.enable_adaptation == True:
                    avg_data_collection = evaluate(self.batch_train_agent.batch_agent[tidx])
                    model_idx = self.batch_train_agent.corresponding_model[tidx]
                    for it in range(self.prm.snap_iter_nums):
                        update_policy = (it % self.policy_freq == 0)
                        n_samples = self.batch_train_agent.batch_agent[tidx].replay_buffer.size()
                        actor_complex_term = get_task_complexity(self.prm, self.actor_prior_model, self.actor_posteriors_train_models[model_idx], n_samples, noised_prior=False)
                        critic_complex_term = get_task_complexity(self.prm, self.critic_prior_model, self.critic_posteriors_train_models[model_idx], n_samples,  noised_prior=False)
                        self.batch_train_agent.batch_agent[tidx].adaptation(actor_complex_term, critic_complex_term, update_policy)
                    dc_rewards.append(avg_data_collection)

                ############# ############### #############
                avg_reward = evaluate(self.batch_train_agent.batch_agent[tidx])
                all_task_rewards.append(avg_reward)
        elif mode == 'eval':
            # Step 1: Reset the model
            for i in range(len(eval_tasks)):
                self.actor_posteriors_eval_models[i].load_state_dict(self.actor_prior_model.state_dict())
                self.critic_posteriors_eval_models[i].load_state_dict(self.critic_prior_model.state_dict())
                self.batch_eval_agent.batch_agent[i].reset_model()
            for i, tidx in enumerate(eval_tasks):
                ############# adaptation step #############
                if self.prm.enable_adaptation == True:
                    avg_data_collection = evaluate(self.batch_eval_agent.batch_agent[tidx])
                    model_idx = self.batch_eval_agent.corresponding_model[tidx]
                    for it in range(self.prm.snap_iter_nums):
                        update_policy = (it % self.policy_freq == 0)
                        n_samples = self.batch_eval_agent.batch_agent[tidx].replay_buffer.size()
                        actor_complex_term = get_task_complexity(self.prm, self.actor_prior_model, self.actor_posteriors_eval_models[model_idx], n_samples)
                        critic_complex_term = get_task_complexity(self.prm, self.critic_prior_model, self.critic_posteriors_eval_models[model_idx], n_samples)
                        self.batch_eval_agent.batch_agent[tidx].adaptation(actor_complex_term, critic_complex_term, update_policy)
                    dc_rewards.append(avg_data_collection)
                ############# ############### #############
                avg_reward = evaluate(self.batch_eval_agent.batch_agent[tidx])
                all_task_rewards.append(avg_reward)
        else:
            raise ValueError

        ############# adaptation step #############
        # Roll-back
        ############# ############### #############
        if self.prm.enable_adaptation == True:
            self.rollback(eval_tasks, mode)

        if self.prm.enable_adaptation == True:
            msg += ' *** with Adapation *** '
            print('Avg rewards (only one eval loop) for all tasks before adaptation ',
                  np.mean(dc_rewards))
        print("---------------------------------------")
        print("%s over %d episodes of %d %s tasks :%f" \
              % (msg, self.prm.num_evals, len(self.meta_eval_tasks), mode, np.mean(all_task_rewards)))
        print("---------------------------------------")

        return np.mean(all_task_rewards)

    def rollout_policy(self, agent):
        agent.reset_env()

        trajectory = []
        total_reward = 0

        state = agent.env.reset()
        done = False
        step = 0

        while not done and step < self.prm.max_path_length:
            action = agent.select_action(state)
            next_state, reward, done, _ = agent.env.step(action)

            reward = np.array(reward, dtype=np.float32)
            total_reward += reward
            trajectory.append((state, action, reward, next_state, done))

            state = next_state
            step += 1

        return trajectory, total_reward

    def rollback(self, task_list, mode='eval'):
        self.actor_prior_model.load_state_dict(self.actor_prior_model_copy.state_dict())
        self.critic_prior_model.load_state_dict(self.critic_prior_model_copy.state_dict())

        if mode == 'train':
            for i_task, task_idx in enumerate(task_list):
                model_idx = self.batch_train_agent.corresponding_model[task_idx]
                self.actor_posteriors_train_models[model_idx].load_state_dict(self.actor_posteriors_train_models_copy[model_idx].state_dict())
                self.critic_posteriors_train_models[model_idx].load_state_dict(self.critic_posteriors_train_models_copy[model_idx].state_dict())

                self.batch_train_agent.batch_agent[task_idx].rollback()
        elif mode == 'eval':
            for i_task, task_idx in enumerate(task_list):
                model_idx = self.batch_eval_agent.corresponding_model[task_idx]
                self.actor_posteriors_eval_models[model_idx].load_state_dict(self.actor_posteriors_eval_models_copy[model_idx].state_dict())
                self.critic_posteriors_eval_models[model_idx].load_state_dict(self.critic_posteriors_eval_models_copy[model_idx].state_dict())

                self.batch_eval_agent.batch_agent[task_idx].rollback()
        else:
            raise ValueError

    def save_model_states(self, mode='eval'):
        ####### ####### ####### Super Important ####### ####### #######
        # Step 0: It is very important to make sure that we save model params before
        # do anything here
        ####### ####### ####### ####### ####### ####### ####### #######
        if not hasattr(self, 'actor_prior_model_copy'):
            self.actor_prior_model_copy = deepcopy(self.actor_prior_model)
            self.critic_prior_model_copy = deepcopy(self.critic_prior_model)
        else:
            self.actor_prior_model_copy.load_state_dict(self.actor_prior_model.state_dict())
            self.critic_prior_model_copy.load_state_dict(self.critic_prior_model.state_dict())

        if mode == 'train':
            if not hasattr(self, 'actor_posteriors_train_models_copy'):
                self.actor_posteriors_train_models_copy = [deepcopy(actor_posteriors_train_model) for
                                                           actor_posteriors_train_model in
                                                           self.actor_posteriors_train_models]
                self.critic_posteriors_train_models_copy = [deepcopy(critic_posteriors_eval_model) for
                                                            critic_posteriors_eval_model in
                                                            self.critic_posteriors_train_models]
            else:
                for i in range(len(self.meta_train_tasks)):
                    self.actor_posteriors_train_models_copy[i].load_state_dict(
                        self.actor_posteriors_train_models[i].state_dict())
                    self.critic_posteriors_train_models_copy[i].load_state_dict(
                        self.critic_posteriors_train_models[i].state_dict())
            n_tasks = len(self.meta_train_tasks)
            for i_task in range(n_tasks):
                self.batch_train_agent.batch_agent[i_task].save_model_states()

        elif mode == 'eval':
            if not hasattr(self, 'actor_posteriors_eval_models_copy'):
                self.actor_posteriors_eval_models_copy = [deepcopy(actor_posteriors_eval_model) for
                                                          actor_posteriors_eval_model in
                                                          self.actor_posteriors_eval_models]
                self.critic_posteriors_eval_models_copy = [deepcopy(critic_posteriors_eval_model) for
                                                           critic_posteriors_eval_model in
                                                           self.critic_posteriors_eval_models]
            else:
                for i in range(len(self.meta_eval_tasks)):
                    self.actor_posteriors_eval_models_copy[i].load_state_dict(
                        self.actor_posteriors_eval_models[i].state_dict())
                    self.critic_posteriors_eval_models_copy[i].load_state_dict(
                        self.critic_posteriors_eval_models[i].state_dict())
            n_tasks = len(self.meta_eval_tasks)
            for i_task in range(n_tasks):
                self.batch_eval_agent.batch_agent[i_task].save_model_states()

        else:
            raise ValueError

    def collect_data_for_adaptation(self, eval_agent):
        """
            Collect data for adaptation.
        """
        eval_agent.empty_buffer()

        reward_info = []
        total_step = 0

        while total_step < self.snap_init_steps:
            trajectory, traj_reward = self.rollout_policy(eval_agent)
            eval_agent.add_trajectory(trajectory)
            total_step += len(trajectory)
            reward_info.append(traj_reward)

        eval_agent.normalize_states()

        return np.mean(reward_info)




