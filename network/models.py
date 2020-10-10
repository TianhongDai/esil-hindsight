import torch
from torch import nn
from torch.nn import functional as F

"""
this network also include gaussian distribution

It has been revised for the goal-orientation tasks

"""

class MLP_Net(nn.Module):
    def __init__(self, state_size, num_actions, dist_type):
        super(MLP_Net, self).__init__()
        self.dist_type = dist_type
        # define the critic network
        self.fc1_v = nn.Linear(state_size, 256)
        self.fc2_v = nn.Linear(256, 256)
        self.fc3_v = nn.Linear(256, 256)
        # define the actor network
        self.fc1_a = nn.Linear(state_size, 256)
        self.fc2_a = nn.Linear(256, 256)
        self.fc3_a = nn.Linear(256, 256)
        # check the type of distribution
        if self.dist_type == 'gauss':
            self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))
            self.action_mean = nn.Linear(256, num_actions)
            self.action_mean.weight.data.mul_(0.1)
            self.action_mean.bias.data.zero_()
        elif self.dist_type == 'beta':
            raise NotImplementedError
        # define layers to output state value
        self.value = nn.Linear(256, 1)
        self.value.weight.data.mul_(0.1)
        self.value.bias.data.zero_()

    def forward(self, x):
        x_v = F.relu(self.fc1_v(x))
        x_v = F.relu(self.fc2_v(x_v))
        x_v = F.relu(self.fc3_v(x_v))
        state_value = self.value(x_v)
        # output the policy...
        x_a = F.relu(self.fc1_a(x))
        x_a = F.relu(self.fc2_a(x_a))
        x_a = F.relu(self.fc3_a(x_a))
        if self.dist_type == 'gauss':
            mean = self.action_mean(x_a)
            sigma_log = self.sigma_log.expand_as(mean)
            sigma = torch.exp(sigma_log)
            pi = (mean, sigma)
        else:
            raise NotImplementedError
        return state_value, pi
