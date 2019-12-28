# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
import os
import sys
import math
from collections import deque, namedtuple

import torch
import random
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision.transforms import Compose, CenterCrop, \
    Grayscale, Resize, ToPILImage, ToTensor


from utils.helpers import get_logger

logger = get_logger(__file__)


class ControlNet(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, lr, device):

    super(ControlNet, self).__init__()

    self.lr = lr
    self.device = device
    self.input_shape = input_shape
    self.state_size = state_size
    self.action_size = action_size

    self.fc1 = nn.Linear(input_shape[0] * state_size, 128)
    self.action = nn.Linear(128, action_size)
    self.value = nn.Linear(128, 1)

    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):

    x = x.to(self.device).float()

    x = F.relu(self.fc1(x))
    q = self.action(x)
    v = self.value(x)
    aprob = self.softmax(q)

    # action, log_prob for action, value estimate for action, softmax probs
    return aprob, v


class AgentOfControl():

  def __init__(self, cfgs, action_size=None, device=None, model_file=None):

    self.history = None
    self.rewards = None
    self.values = None
    self.aprobs = None
    self.log_probs = None
    self.batch_loss = []
    self.batch_rewards = []
    self.input_shape = cfgs['input_shape']
    self.lr = cfgs['lr']
    self.gamma = cfgs['gamma']
    self.state_size = cfgs['state_size']
    self.action_size = action_size
    self.device = device

    assert self.input_shape is not None, 'Input shape has to be not None'
    assert self.action_size is not None, 'Action size has to non None'
    assert self.device is not None, 'Device has to be CPU/GPU'

    self.zero_state = torch.zeros([1] + self.input_shape, dtype=torch.float)

    self.policy = ControlNet(self.input_shape, self.state_size,
                             self.action_size, self.lr,
                             self.device).to(self.device)

    self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    if model_file:
      self.load_model(model_file)

  def reset(self):

    self.batch_loss = []
    self.batch_rewards = []

    self.flash_episode()

  def flash_episode(self):

    self.rewards = []
    self.log_probs = []
    self.values = []
    self.aprobs = []

    no_history = [self.zero_state for _ in range(self.state_size)]
    self.history = deque(no_history, maxlen=self.state_size)

  def load_model(self, model_file):

    logger.info('Loading agent weights from {}'.format(model_file))

    self.policy.load_state_dict(torch.load(model_file))
    self.policy.eval()

  def get_action(self, state):

    aprob, value = self.policy(state)

    c = Categorical(aprob)
    a = c.sample()

    log_prob = c.log_prob(a)

    self.log_probs.append(log_prob)
    self.values.append(value[0])
    self.aprobs.append(aprob[0])

    return a.detach().cpu().numpy()[0]

  def append_state(self, state):

    state = torch.from_numpy(state)

    self.history.append(state)

  def get_state(self):

    return torch.cat(list(self.history)).unsqueeze(0)

  def append_reward(self, r):

    self.rewards.append(r)

  def get_episode_rewards(self):

    return np.sum(self.rewards)

  def discount_episode(self):

    ep_length = len(self.rewards)
    gamma = torch.tensor(self.gamma)
    ep_rewards = torch.tensor(self.rewards, dtype=torch.float32)
    ep_discounts = torch.tensor([gamma ** t for t in range(ep_length)],
                                dtype=torch.float32)

    rewards = [ep_rewards[idx:] * ep_discounts[:ep_length - idx]
               for idx in range(ep_length)]

    rewards = list(map(torch.sum, rewards))
    rewards = torch.stack(rewards).to(self.device)

    mean, std = rewards.mean(), rewards.std()
    rewards = (rewards - mean)/(std + np.finfo(np.float32).eps.item())

    neg_log_probs = torch.cat(self.log_probs).mul(-1)
    policy_loss = neg_log_probs * rewards

    self.batch_loss.append(policy_loss)

  def append_episode_reward(self, ep_reward):

    self.batch_rewards.append(ep_reward)

  def optimize(self):

    self.optimizer.zero_grad()
    batch_policy_loss = torch.cat(self.batch_loss).mean()
    batch_policy_loss.backward()
    self.optimizer.step()

    return batch_policy_loss.detach().cpu().numpy()

  def save_model(self, step, dest):

    model_savefile = '{0}/atari-agent-{1}.pth'.format(dest, step)
    logger.debug("Saving Atari Agent to {}".format(model_savefile))

    torch.save(self.policy.state_dict(), model_savefile)
