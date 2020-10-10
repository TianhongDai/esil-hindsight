import argparse

"""
arguments for the ESIL
"""

def get_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--gamma', type=float, default=0.98, help='the discount factor of RL')
    parse.add_argument('--seed', type=int, default=153, help='the random seeds')
    parse.add_argument('--ncycles', type=int, default=50, help='the cycles to collect samples')
    parse.add_argument('--env-name', type=str, default='FetchReach-v1', help='the environment name')
    parse.add_argument('--batch-size', type=int, default=20, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=3e-4, help='learning rate of the algorithm')
    parse.add_argument('--epoch', type=int, default=10, help='the epoch during training')
    parse.add_argument('--vloss-coef', type=float, default=1, help='the coefficient of value loss')
    parse.add_argument('--ent-coef', type=float, default=0, help='the entropy loss coefficient')
    parse.add_argument('--tau', type=float, default=0.95, help='gae coefficient')
    parse.add_argument('--cuda', action='store_true', help='use cuda do the training')
    parse.add_argument('--total-frames', type=int, default=10000000, help='the total frames for training')
    parse.add_argument('--dist', type=str, default='gauss', help='the distributions for sampling actions')
    parse.add_argument('--eps', type=float, default=1e-5, help='param for adam optimizer')
    parse.add_argument('--clip', type=float, default=0.2, help='the ratio clip param')
    parse.add_argument('--save-dir', type=str, default='saved_models', help='the folder to save models')
    parse.add_argument('--lr-decay', action='store_true', help='if using the learning rate decay during decay')
    parse.add_argument('--max-grad-norm', type=float, default=0.5, help='grad norm')
    parse.add_argument('--display-interval', type=int, default=10, help='the interval that display log information')
    parse.add_argument('--clip-range', type=float, default=5, help='the clip range when normalize the observation')
    parse.add_argument('--n-test-rollouts', type=int, default=10, help='numbers to do the evaluation')
    parse.add_argument('--clip-obs', type=float, default=200, help='clip the observation')
    parse.add_argument('--render', action='store_true', help='render the demo')
    # following are the parameters of the sil module
    parse.add_argument('--max-nlogp', type=float, default=5, help='max nlogp')
    parse.add_argument('--demo-length', type=int, default=10, help='the demo length')
    parse.add_argument('--dist-per-step', type=float, default=0.01, help='the minimum step per step')
    parse.add_argument('--alpha', type=float, default=1, help='the minimum step per step')
    parse.add_argument('--beta', type=float, default=1, help='the minimum step per step')
    parse.add_argument('--adaptive-beta', action='store_true', help='if use the adaptive clone beta coefficient')

    args = parse.parse_args()

    return args
