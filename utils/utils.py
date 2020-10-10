import numpy as np
import torch
from torch.distributions.normal import Normal
import random

def select_actions(pi, dist_type):
    if dist_type == 'gauss':
        mean, std = pi
        actions = Normal(mean, std).sample()
    else:
        raise NotImplementedError
    # return actions
    return actions.detach().cpu().numpy().squeeze()

# evaluate the actions
def evaluate_actions(pi, actions, dist_type):
    if dist_type == 'gauss':
        mean, std = pi
        normal_dist = Normal(mean, std)
        log_prob = normal_dist.log_prob(actions).sum(dim=1, keepdim=True)
        entropy = normal_dist.entropy().mean()
    else:
        raise NotImplementedError
    return log_prob, entropy

# get the environment's parameters
def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0],
            'goal': obs['desired_goal'].shape[0],
            'action': env.action_space.shape[0],
            'action_max': env.action_space.high[0],
            }
    params['max_timesteps'] = env._max_episode_steps
    return params
