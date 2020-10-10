from arguments import get_args
import numpy as np
from network.models import MLP_Net
from utils.utils import get_env_params
import torch
import os, gym

"""
script to watch the demo of the ESIL
"""

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # create environment
    env = gym.make(args.env_name)
    # get the environment parameters
    env_params = get_env_params(env)
    # start to create model
    model_path = '{}/{}/model.pt'.format(args.save_dir, args.env_name)
    network = MLP_Net(env_params['obs'] + env_params['goal'], env_params['action'], args.dist)
    network_model, obs_mean, obs_std, g_mean, g_std = torch.load(model_path, map_location='cpu')
    network.load_state_dict(network_model)
    network.eval()
    # start to do the testing
    for i in range(args.demo_length):
        observation = env.reset()
        # start to do the demo
        obs, g = observation['observation'], observation['desired_goal']
        for t in range(env._max_episode_steps):
            if args.render:
                env.render()
            inputs = process_inputs(obs, g, obs_mean, obs_std, g_mean, g_std, args)
            with torch.no_grad():
                _, pi = network(inputs)
                if args.dist == 'gauss':
                    mean, std = pi
                    input_actions = mean.detach().cpu().numpy().squeeze()
                else:
                    raise NotImplementedError
            # put actions into the environment
            observation_new, reward, _, info = env.step(input_actions)
            obs = observation_new['observation']
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
