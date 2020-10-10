from arguments import get_args
from rl_base.ppo_agent import ppo_agent
from network.models import MLP_Net
from utils.utils import get_env_params
import gym, os, random
import numpy as np
import torch
from mpi4py import MPI

"""
script to train the agent with ESIL
"""

if __name__ == '__main__':
    # set some environment variables
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'
    # get the arguments
    args = get_args()
    # start to set the random seeds
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # make environment
    env = gym.make(args.env_name)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    # get the environment parameters
    env_params = get_env_params(env)
    network = MLP_Net(env_params['obs'] + env_params['goal'], env_params['action'], args.dist)
    ppo_trainer = ppo_agent(env, args, network, env_params)
    ppo_trainer.learn()
