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

  def forward(self, x):

    x = x.to(self.device).float()

    x = F.relu(self.fc1(x))
    q = self.action(x)
    v = self.value(x)

    # logits, value estimate for state
    return q, v


class AgentOfControl():

  def __init__(self, cfgs, action_size=None, device=None, model_file=None):

    self.history = None
    self.states = None
    self.action = None
    self.rewards = None
    self.values = None
    self.mb_states = None
    self.mb_actions = None
    self.mb_rewards = None
    self.mb_gaes = None
    self.ep_rewards = []
    self.gamma = cfgs['gamma']
    self.lambd = cfgs['lambda']
    self.policy_lr = cfgs['policy_lr']
    self.value_lr = cfgs['value_lr']
    self.value_iter = cfgs['value_iter']
    self.state_size = cfgs['state_size']
    self.input_shape = cfgs['input_shape']
    self.action_size = action_size
    self.device = device

    assert self.input_shape is not None, 'Input shape has to be not None'
    assert self.action_size is not None, 'Action size has to non None'
    assert self.device is not None, 'Device has to be CPU/GPU'

    self.zero_state = torch.zeros([1] + self.input_shape, dtype=torch.float)

    self.policy = ControlNet(self.input_shape, self.state_size,
                             self.action_size, self.policy_lr,
                             self.device).to(self.device)
    self.value = ControlNet(self.input_shape, self.state_size,
                            self.action_size, self.value_lr,
                            self.device).to(self.device)

    self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                       lr=self.policy_lr)
    self.value_optimizer = optim.Adam(self.value.parameters(),
                                      lr=self.value_lr)

    if model_file:
      self.load_model(model_file)

  def reset(self):

    self.mb_states = []
    self.mb_actions = []
    self.mb_rewards = []
    self.mb_gaes = []
    self.ep_rewards = []

    self.flash_episode()

  def flash_episode(self):

    self.states = []
    self.actions = []
    self.rewards = []
    self.values = []

    no_history = [self.zero_state for _ in range(self.state_size)]
    self.history = deque(no_history, maxlen=self.state_size)

  def load_model(self, model_file):

    logger.info('Loading agent weights from {}'.format(model_file))

    self.policy.load_state_dict(torch.load(model_file))
    self.policy.eval()

  def get_action(self, state, deterministic=False):

    with torch.no_grad():
      logits, _ = self.policy(state)
      _, value = self.value(state)

    if deterministic:
      return logits.max(1)[1]

    c = Categorical(logits=logits)
    a = c.sample()

    self.states.append(state)
    self.actions.append(a)
    self.values.append(value)

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

  def discount(self, episode, cr, normalize=False):

    ep_length = episode.shape[0]

    if ep_length == 1:
      # std is not defined for array of length 1
      return

    gs = torch.stack([cr.pow(t) for t in range(ep_length)])

    ep_discounted = [episode[idx:] * gs[:ep_length - idx]
                     for idx in range(ep_length)]

    episode = list(map(torch.sum, ep_discounted))
    episode = torch.stack(episode)

    if normalize:

      mean, std = episode.mean(), episode.std()
      episode = (episode - mean)/(std + np.finfo(np.float32).eps.item())

    return episode

  def discount_episode(self):

    gamma = torch.tensor(self.gamma).to(self.device)
    lambd = torch.tensor(self.lambd).to(self.device)

    vals = torch.cat(self.values).squeeze(1)
    rews = torch.tensor(self.rewards, dtype=torch.float32).to(self.device)

    # GAE-lambda estimate via TD
    td = rews[:-1] + gamma * vals[1:] - vals[:-1]

    gaes = self.discount(td, gamma * lambd, normalize=True)
    rewards = self.discount(rews, gamma)[:-1]

    states = torch.cat(self.states)[:-1]
    actions = torch.cat(self.actions)[:-1]

    self.mb_states.append(states)
    self.mb_actions.append(actions)
    self.mb_rewards.append(rewards)
    self.mb_gaes.append(gaes)

  def append_episode_reward(self, ep_reward):

    self.ep_rewards.append(ep_reward)

  def optimize(self):

    mb_states = torch.cat(self.mb_states)
    mb_actions = torch.cat(self.mb_actions)
    mb_rewards = torch.cat(self.mb_rewards)
    mb_gaes = torch.cat(self.mb_gaes)

    # policy optimisation
    self.policy_optimizer.zero_grad()
    mb_logits, _ = self.policy(mb_states)
    ce = F.cross_entropy(mb_logits, mb_actions, reduction='none')
    policy_loss = (mb_gaes * ce).mean()
    policy_loss.backward()
    self.policy_optimizer.step()

    # value optimisation
    self.value_optimizer.zero_grad()
    _, mb_values = self.value(mb_states)
    mb_values = mb_values.squeeze(1)
    value_loss = F.smooth_l1_loss(mb_values, mb_rewards)
    value_loss.backward()
    self.value_optimizer.step()

    loss = policy_loss + value_loss

    return loss.detach().cpu().numpy()
