import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
import csv


def weights_init_(m):
    """Initialize Weights of Critic and Actor network"""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def var(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


def get_tensor(arr):
    if len(arr.shape) == 1:
        return var(torch.FloatTensor(arr.copy())).unsqueeze(0)
    else:
        return var(torch.FloatTensor(arr.copy()))


def _is_update(step, freq, ignore=0, rem=0):
    if step != ignore and step % freq == rem:
        return True
    return False


def listdirs(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]


def record_experience_to_csv(config, experiment_name, csv_name='experiments.csv'):
    # append DATE_TIME to dict
    d = config
    d['date'] = experiment_name

    if os.path.exists(csv_name):
        # Save Dictionary to a csv
        with open(csv_name, 'a') as f:
            w = csv.DictWriter(f, list(d.keys()))
            w.writerow(d)
    else:
        # Save Dictionary to a csv
        with open(csv_name, 'w') as f:
            w = csv.DictWriter(f, list(d.keys()))
            w.writeheader()
            w.writerow(d)


class SubgoalActionSpace(object):
    def __init__(self, dim=120):
        limits = np.array([0.0] * 120)
        self.shape = (dim, 1)
        self.low = limits[:dim]
        self.high = np.r_[[1.3]*60, [1.0]*60]

    def sample(self):
        return (self.high - self.low) * np.random.sample() + self.low


class Subgoal(object):
    def __init__(self, dim=120):
        self.action_space = SubgoalActionSpace(dim)
        self.action_dim = self.action_space.shape[0]


class Logger:
    def __init__(self, log_path):
        self.writer = SummaryWriter(log_path)

    def print(self, name, value, episode=-1, step=-1):
        string = "{} is {}".format(name, value)
        if episode > 0:
            print('Episode:{}, {}'.format(episode, string))
        if step > 0:
            print('Step:{}, {}'.format(step, string))

    def write(self, name, value, index):
        self.writer.add_scalar(name, value, index)