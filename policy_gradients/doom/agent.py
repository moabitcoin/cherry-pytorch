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


class DoomNet(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, device):

    super(DoomNet, self).__init__()

    self.device = device
    self.input_shape = input_shape
    self.state_size = state_size
    self.action_size = action_size

    (w, h) = self.input_shape

    self.conv1 = nn.Conv2d(self.state_size, 16, kernel_size=8, stride=4)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=8, stride=4)

    def feat_shape(size, kernel_size=8, stride=4):
      return (size - (kernel_size - 1) - 1) // stride + 1
    convw = feat_shape(feat_shape(w))
    convh = feat_shape(feat_shape(h))
    feat_spatial_shape = convw * convh * 32

    self.head = nn.Linear(feat_spatial_shape, 256)
    self.action = nn.Linear(256, self.action_size)
    self.value = nn.Linear(256, 1)

    self.log_softmax = nn.LogSoftmax(dim=1)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.0)

    if type(m) == nn.Conv2d:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.0)

  def forward(self, x):

    x = x.to(self.device).float() / 255.

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.head(x))

    q = self.action(x)
    v = self.value(x)

    # logits, value estimate for state
    return q, v


class AgentOfDoom():

  def __init__(self, cfgs, action_size=None, device=None, model_file=None):

    self.history = None
    self.states = None
    self.actions = None
    self.rewards = None
    self.values = None
    self.mb_states = None
    self.mb_actions = None
    self.mb_rewards = None
    self.mb_values = None
    self.ep_rewards = None
    self.gamma = cfgs['gamma']
    self.policy_lr = cfgs['policy_lr']
    self.value_lr = cfgs['value_lr']
    self.state_size = cfgs['state_size']
    self.crop_shape = cfgs['crop_shape']
    self.input_shape = cfgs['input_shape']
    self.action_size = action_size
    self.device = device

    assert self.input_shape is not None, 'Input shape has to be not None'
    assert self.action_size is not None, 'Action size has to non None'
    assert self.device is not None, 'Device has to be CPU/GPU'

    self.zero_state = torch.zeros([1] + self.input_shape, dtype=torch.uint8)

    transforms = [ToPILImage()]
    if self.crop_shape:
      transforms.append(CenterCrop(self.crop_shape))
    transforms.append(Resize(self.input_shape))

    self.transform = Compose(transforms)

    self.policy = DoomNet(self.input_shape, self.state_size,
                          self.action_size, self.device).to(self.device)

    self.value = DoomNet(self.input_shape, self.state_size,
                         self.action_size, self.device).to(self.device)

    # self.policy.apply(self.policy.init_weights)

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
    self.mb_values = []
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

  def eval(self):

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

    state = self.zero_state if state is None else state

    state = np.array(self.transform(state), dtype=np.uint8)

    state = torch.from_numpy(state).view(1, self.input_shape[0],
                                         self.input_shape[1])

    self.history.append(state)

  def set_state(self, state):

    for _ in range(self.state_size):
      self.append_state(state)

  def append_reward(self, r):

    self.rewards.append(r)

  def get_state(self):

    return torch.cat(list(self.history)).unsqueeze(0)

  def discount_episode(self):

    ep_length = len(self.rewards)
    ep_rewards = torch.tensor(self.rewards, dtype=torch.float32,
                              requires_grad=False)
    ep_discounts = torch.tensor([self.gamma ** t for t in range(ep_length)],
                                dtype=torch.float32, requires_grad=False)

    rewards = [ep_rewards[idx:] * ep_discounts[:ep_length - idx]
               for idx in range(ep_length)]

    rewards = list(map(torch.sum, rewards))

    states = torch.cat(self.states)
    actions = torch.cat(self.actions)
    rewards = torch.stack(rewards).to(self.device)
    values = torch.cat(self.values)

    mean, std = rewards.mean(), rewards.std()
    rewards = (rewards - mean)/(std + np.finfo(np.float32).eps.item())

    self.mb_states.append(states)
    self.mb_actions.append(actions)
    self.mb_rewards.append(rewards)
    self.mb_values.append(values)

  def append_episode_reward(self, reward):

    self.ep_rewards.append(reward)

  def get_episode_rewards(self):

    return np.sum(self.rewards)

  def optimize(self):

    mb_states = torch.cat(self.mb_states)
    mb_actions = torch.cat(self.mb_actions)
    mb_rewards = torch.cat(self.mb_rewards)
    mb_values = torch.cat(self.mb_values)

    # policy optimisation
    self.policy_optimizer.zero_grad()
    mb_logits, _ = self.policy(mb_states)
    ce = F.cross_entropy(mb_logits, mb_actions, reduction='none')
    policy_loss = ((mb_rewards - mb_values) * ce).mean()
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
