from src.alogs.agent import TD3_BC


class MultiTaskAgent(object):
    def __init__(self,
                 prm,
                 env,
                 batch_actor_model,
                 batch_critic_model,
                 batch_tasks,
                 multi_replay_buffer,
                 ):

        self.prm = prm
        self.env = env
        self.multi_replay_buffer = multi_replay_buffer
        self.n_batch_model = len(batch_actor_model)

        max_action = float(env.action_space.high[0])

        batch_agent = []
        corresponding_model = []
        for model_idx, task_idx in enumerate(batch_tasks):
            batch_agent.append((task_idx, TD3_BC(prm,
                                                 actor_model=batch_actor_model[model_idx],
                                                 critic_model=batch_critic_model[model_idx],
                                                 multi_buffer=multi_replay_buffer,
                                                 env=env,
                                                 task=task_idx,
                                                 max_action=max_action,
                                                 tau=prm.tau,
                                                 gamma=prm.gamma,
                                                 policy_noise=prm.policy_noise * max_action,
                                                 noise_clip=prm.noise_clip * max_action,
                                                 alpha=prm.alpha
                                                 )))
            corresponding_model.append((task_idx, model_idx))
        self.batch_agent = dict(batch_agent)
        self.corresponding_model = dict(corresponding_model)

    def reset_model(self):
        for key, value in self.batch_agent.items():
            self.batch_agent[key].reset_model()

    def syn_weight(self):
        for key, value in self.batch_agent.items():
            self.batch_agent[key].syn_weight()

    def normalize_states(self):
        for key, value in self.batch_agent.items():
            self.batch_agent[key].normalize_states()
