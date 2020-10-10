import numpy as np
import torch
from torch import optim
from utils.utils import select_actions, evaluate_actions
from utils.mpi_normalizer import normalizer
from utils.her_module import her_module
from utils.mpi_utils import sync_networks, sync_grads
from datetime import datetime
import os, copy
from mpi4py import MPI

"""
mpi version - revised for solving the Fetch Robotics task.

Tianhong Dai

"""

class ppo_agent:
    def __init__(self, envs, args, net, env_params):
        self.envs = envs 
        self.args = args
        # get the environment parameters
        self.env_params = env_params
        # define the newtork...
        self.net = net
        sync_networks(self.net)
        self.old_net = copy.deepcopy(self.net)
        # if use the cuda...
        if self.args.cuda:
            self.net.cuda()
            self.old_net.cuda()
        # define the optimizer...
        self.optimizer = optim.Adam(self.net.parameters(), self.args.lr, eps=self.args.eps)
        # check saving folder..
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path, exist_ok=True)
        # define the global normalizer
        self.o_norm = normalizer(self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(self.env_params['goal'], default_clip_range=self.args.clip_range)
        # set the her module
        self.her = her_module(self.envs.compute_reward)

    # start to train the network...
    def learn(self):
        num_updates = self.args.total_frames // (self.args.ncycles * self.env_params['max_timesteps'])
        # load the self-imitation module
        for update in range(num_updates):
            if self.args.lr_decay:
                self._adjust_learning_rate(update, num_updates)
            mb_obs, mb_actions, mb_dones, mb_g, mb_ag, mb_prob, mb_rewards = [], [], [], [], [], [], []
            for cycle in range(self.args.ncycles):
                resample_needed = True
                # reset the environments
                while resample_needed:
                    ep_obs, ep_actions, ep_dones, ep_g, ep_ag, ep_rewards = [], [], [], [], [], []
                    observation = self.envs.reset()
                    obs, ag, g = observation['observation'], observation['achieved_goal'], observation['desired_goal']
                    # start to collect samples
                    last_pos = ag
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            # get tensors
                            obs_tensor = self._get_tensors(obs, g)
                            _, pis = self.net(obs_tensor)
                        # select actions
                        actions = select_actions(pis, self.args.dist)
                        if self.args.dist == 'gauss':
                            input_actions = actions.copy()
                        else:
                            raise NotImplementedError
                        # start to excute the actions in the environment
                        observation_new, rewards, dones, _ = self.envs.step(input_actions)
                        # get the new observation and achieved goal
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # start to store information - copy() can be removed to save memory
                        ep_obs.append(obs.copy())
                        ep_actions.append(actions.copy())
                        # if it was in the last timestep, make done as True, easy for calcuate the returns
                        if t == (self.env_params['max_timesteps'] - 1):
                            dones = True
                        # calculate the distance change..
                        if self._calculate_displacement(last_pos, ag_new) > self.args.dist_per_step:
                            resample_needed = False
                            last_pos = ag_new
                        ep_dones.append(dones)
                        ep_g.append(g.copy())
                        ep_ag.append(ag.copy())
                        ep_rewards.append(rewards)
                        # reassign the observation and achieved goal
                        obs, ag = obs_new, ag_new
                    ep_ag.append(ag.copy())
                mb_obs.append(ep_obs)
                mb_g.append(ep_g)
                mb_ag.append(ep_ag)
                mb_actions.append(ep_actions)
                mb_dones.append(ep_dones)
                mb_rewards.append(ep_rewards)
            # process the rollouts
            mb_obs = np.asarray(mb_obs, dtype=np.float32)
            mb_g = np.asarray(mb_g, dtype=np.float32)
            mb_ag = np.asarray(mb_ag, dtype=np.float32)
            mb_actions = np.asarray(mb_actions, dtype=np.float32)
            mb_dones = np.asarray(mb_dones, dtype=np.bool)
            mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
            mb_g_her, mb_rewards_her = self.her.update_trajectory(mb_g, mb_ag)
            # reshape the trajectory data and continue the processing
            mb_obs = mb_obs.reshape(-1, mb_obs.shape[2])
            mb_g = mb_g.reshape(-1, mb_g.shape[2])
            mb_g_her = mb_g_her.reshape(-1, mb_g_her.shape[2])
            mb_actions = mb_actions.reshape(-1, mb_actions.shape[2])
            mb_rewards = mb_rewards.flatten()
            mb_rewards_her = mb_rewards_her.flatten()
            mb_dones = mb_dones.flatten()
            # start to compute the return
            mb_returns = np.zeros_like(mb_rewards)
            mb_returns_her = np.zeros_like(mb_rewards)
            # not use the GAE in this time...
            for t in reversed(range(int(self.args.ncycles * self.env_params['max_timesteps']))):
                if t == int(self.args.ncycles * self.env_params['max_timesteps']) - 1:
                    # it doesn't matter here...because Fetch is fixed length environment
                    nextnonterminal = 0
                    nextvalues = 0
                    nextvalues_her = 0
                else:
                    nextnonterminal = 1.0 - mb_dones[t]
                    nextvalues = mb_returns[t + 1]
                    nextvalues_her = mb_returns_her[t + 1]
                # calculate the returns
                mb_returns[t] = mb_rewards[t] + self.args.gamma * nextvalues * nextnonterminal
                mb_returns_her[t] = mb_rewards_her[t] + self.args.gamma * nextvalues_her * nextnonterminal
            # before update the network, the old network will try to load the weights
            self.old_net.load_state_dict(self.net.state_dict())
            # start to update the network
            pl, vl, ent, clone_samples = self._update_network(mb_obs, mb_g, mb_g_her, mb_actions, mb_returns, mb_returns_her)
            if update % self.args.display_interval == 0:
                success_rate = self._eval_agent()
                # display the training information
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print('[{}] Update: {} / {}, Frames: {}, Success: {:.3f}, PL: {:.3f}, '\
                            'VL: {:.3f}, Ent: {:.3f}, SAMPLES: {}'.format(datetime.now(), update, num_updates, \
                        (update + 1)*self.args.ncycles * self.env_params['max_timesteps'], success_rate, pl, vl, ent, clone_samples))
                    # save the model
                    torch.save([self.net.state_dict(), self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std], \
                            '{}/model.pt'.format(self.model_path))

    # update the network
    def _update_network(self, obs, goals, goals_her, actions, returns, returns_her):
        # update the mpi normalizer firstly...
        self.o_norm.update(obs)
        self.o_norm.recompute_stats()
        self.g_norm.update(np.concatenate([goals, goals_her], axis=0))
        self.g_norm.recompute_stats()
        # start to update the network
        inds = np.arange(obs.shape[0])
        nbatch_train = obs.shape[0] // self.args.batch_size
        for _ in range(self.args.epoch):
            np.random.shuffle(inds)
            for start in range(0, obs.shape[0], nbatch_train):
                # get the mini-batchs
                end = start + nbatch_train
                mbinds = inds[start:end]
                mb_obs, mb_g, mb_g_her, mb_actions = obs[mbinds], goals[mbinds], goals_her[mbinds], actions[mbinds]
                mb_returns, mb_returns_her = returns[mbinds], returns_her[mbinds]
                masks = (mb_returns_her > mb_returns).astype(np.float32)
                num_clone_samples = np.sum(masks)
                # normalize the observation and the goal
                mb_obs_norm = self.o_norm.normalize(mb_obs)
                mb_g_norm = self.g_norm.normalize(mb_g)
                mb_g_her_norm = self.g_norm.normalize(mb_g_her)
                # convert minibatches to tensor
                mb_inputs = np.concatenate([mb_obs_norm, mb_g_norm], axis=1)
                mb_inputs_her = np.concatenate([mb_obs_norm, mb_g_her_norm], axis=1)
                # convert the data into the tensor
                mb_inputs = torch.tensor(mb_inputs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
                mb_inputs_her = torch.tensor(mb_inputs_her, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
                mb_actions = torch.tensor(mb_actions, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
                mb_returns = torch.tensor(mb_returns, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
                mb_returns_her = torch.tensor(mb_returns_her, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
                masks = torch.tensor(masks, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(1)
                # normalize adv
                max_nlogp = torch.tensor(np.ones((mb_obs.shape[0], 1)) * (-self.args.max_nlogp), dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu')
                # start to get values
                mb_values, pis = self.net(mb_inputs)
                # start to calculate the value loss...
                value_loss = (mb_returns - mb_values).pow(2).mean()
                # start to calculate the policy loss
                with torch.no_grad():
                    old_values, old_pis = self.old_net(mb_inputs)
                    # get the old log probs
                    old_log_prob, _ = evaluate_actions(old_pis, mb_actions, self.args.dist)
                    old_log_prob = old_log_prob.detach()
                    #old_log_prob = torch.max(old_log_prob, max_nlogp)
                    mb_advs = mb_returns - old_values
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)
                    mb_advs = mb_advs.detach()
                # evaluate the current policy
                log_prob, ent_loss = evaluate_actions(pis, mb_actions, self.args.dist)
                prob_ratio = torch.exp(log_prob - old_log_prob)
                # surr1
                surr1 = prob_ratio * mb_advs
                surr2 = torch.clamp(prob_ratio, 1 - self.args.clip, 1 + self.args.clip) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()
                # final total loss
                ppo_loss = policy_loss + self.args.vloss_coef * value_loss - ent_loss * self.args.ent_coef
                # let's calculate the cloning loss
                _, pis_clone = self.net(mb_inputs_her)
                log_prob_clone, _ = evaluate_actions(pis_clone, mb_actions, self.args.dist)
                log_prob_clone = torch.max(log_prob_clone, max_nlogp)
                # generate the masks
                num_clone_samples = np.max([num_clone_samples, 1])
                esil_loss = -torch.sum(log_prob_clone * masks) / num_clone_samples
                if self.args.adaptive_beta:
                    total_loss = self.args.alpha * ppo_loss + (num_clone_samples / nbatch_train) * esil_loss
                else:
                    total_loss = self.args.alpha * ppo_loss + self.args.beta * esil_loss
                # clear the grad buffer
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.args.max_grad_norm)
                sync_grads(self.net)
                # update
                self.optimizer.step()
        return policy_loss.item(), value_loss.item(), ent_loss.item(), num_clone_samples

    # calcluate the displacement...
    def _calculate_displacement(self, pos1, pos2):
        return np.linalg.norm(pos2 - pos1)

    # convert the numpy array to tensors
    def _get_tensors(self, obs, g):
        obs_norm = self.o_norm.normalize(obs)
        g_norm = self.g_norm.normalize(g)
        # concatenate the stuffs
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32, device='cuda' if self.args.cuda else 'cpu').unsqueeze(0)
        return inputs

    # adjust the learning rate
    def _adjust_learning_rate(self, update, num_updates):
        lr_frac = 1 - (update / num_updates)
        adjust_lr = self.args.lr * lr_frac
        for param_group in self.optimizer.param_groups:
             param_group['lr'] = adjust_lr

    # evaluate the agent..
    def _eval_agent(self):
        total_success_rate = []
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.envs.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    input_tensor = self._get_tensors(obs, g)
                    _, pi = self.net(input_tensor)
                    # convert the actions
                    if self.args.dist == 'gauss':
                        mean, std = pi
                        input_actions = mean.detach().cpu().numpy().squeeze()
                    else:
                        raise NotImplementedError
                observation_new, _, _, info = self.envs.step(input_actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        local_success_rate = np.mean(total_success_rate[:, -1])
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
