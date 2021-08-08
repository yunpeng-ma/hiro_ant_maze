import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from hiro.utils import weights_init_


class SacCritic(nn.Module):
    """Critic Network"""
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim):
        super(SacCritic, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(state_dim + goal_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(state_dim + goal_dim + action_dim, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, s, g, a):
        """ Args: s: state, g: goal, a: action """
        x0 = torch.cat([s, g, a], 1)

        x1 = F.relu(self.linear1(x0))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(x0))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class SacActor(nn.Module):
    """Actor Network"""
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim, action_space=None):
        super(SacActor, self).__init__()

        self.linear1 = nn.Linear(state_dim + goal_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_bound = torch.tensor(1.)
            self.action_shift = torch.tensor(0.)
        else:
            self.action_bound = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_shift = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, s, g):
        x = F.relu(self.linear1(torch.cat([s, g], 1)))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def choose_action_log_prob(self, s, g):
        mean, log_std = self.forward(s, g)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw_action = dist.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        clip_action = torch.tanh(raw_action)
        legal_action = clip_action * self.action_bound + self.action_shift

        log_prob = dist.log_prob(raw_action)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_bound * (1 - clip_action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_bound + self.action_shift
        return legal_action, log_prob, mean

    def to(self, device):
        self.action_bound = self.action_bound.to(device)
        self.action_shift = self.action_shift.to(device)
        return super(SacActor, self).to(device)