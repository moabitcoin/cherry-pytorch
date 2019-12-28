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

    self.conv1 = nn.Conv2d(self.state_size, 16, kernel_size=5,
                           stride=2, bias=False)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, bias=False)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, bias=False)

    def feat_shape(size, kernel_size=5, stride=2):
      return (size - (kernel_size - 1) - 1) // stride + 1
    convw = feat_shape(feat_shape(feat_shape(w)))
    convh = feat_shape(feat_shape(feat_shape(h)))
    feat_spatial_shape = convw * convh * 32

    self.head = nn.Linear(feat_spatial_shape, 256)
    self.action = nn.Linear(256, self.action_size)
    self.value = nn.Linear(256, 1)

    self.softmax = nn.Softmax(dim=-1)

  def init_weights(self, m):
    if type(m) == nn.Linear:
      torch.nn.init.xavier_uniform_(m.weight)
      m.bias.data.fill_(0.0)

    if type(m) == nn.Conv2d:
      torch.nn.init.xavier_uniform_(m.weight)

  def forward(self, x):

    x = x.to(self.device).float() / 255.

    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(x.size(0), -1)
    x = F.relu(self.head(x))

    q = self.action(x)
    v = self.value(x)
    aprob = self.softmax(q)
    c = Categorical(aprob)
    a = c.sample()

    # action, log_prob for action, value estimate for action, softmax probs
    return a, c.log_prob(a), v, aprob


class AgentOfDoom():

  def __init__(self, cfgs, action_size=None, device=None, model_file=None):

    self.history = None
    self.rewards = None
    self.log_probs = None
    self.batch_rewards = None
    self.batch_log_probs = None
    self.episode_scores = None
    self.crop_shape = cfgs['crop_shape']
    self.input_shape = cfgs['input_shape']
    self.lr = cfgs['lr']
    self.gamma = cfgs['gamma']
    self.state_size = cfgs['state_size']
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

    self.policy.apply(self.policy.init_weights)

    self.optimizer = optim.Adam(self.policy.parameters(),
                                lr=self.lr, eps=1.5e-4)

    if model_file:
      self.load_model(model_file)

  def reset(self):

    self.batch_aprobs = []
    self.batch_log_probs = []
    self.batch_rewards = []
    self.batch_values = []
    self.episode_scores = []

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

    action, action_log_prob, value, aprob = self.policy(state)

    self.log_probs.append(action_log_prob)
    self.values.append(value[0])
    self.aprobs.append(aprob[0])

    return action.detach().cpu().numpy()[0]

  def append_state(self, state):

    state = self.zero_state if state is None else state

    state = np.array(self.transform(state), dtype=np.uint8)

    state = torch.from_numpy(state).view(1, self.input_shape[0],
                                         self.input_shape[1])

    self.history.append(state)

  def append_reward(self, r):

    self.rewards.append(r)

  def get_state(self):

    return torch.cat(list(self.history)).unsqueeze(0)

  def get_total_rewards(self):

    num_episodes = len(self.batch_rewards)

    rewards = np.sum(self.episode_scores)
    rewards /= num_episodes

    return rewards

  def discount_episode(self):

    ep_length = len(self.rewards)
    ep_rewards = np.array(self.rewards)
    ep_discounts = np.array([self.gamma ** t for t in range(ep_length)])

    rewards = [ep_rewards[idx:] * ep_discounts[:ep_length - idx]
               for idx in range(ep_length)]

    rewards = np.array(list(map(np.sum, rewards)))

    mean, std = rewards.mean(), rewards.std()
    rewards = (rewards - mean)/(std + np.finfo(np.float32).eps.item())

    rewards = torch.from_numpy(rewards).to(self.device).float()

    log_probs = torch.cat(self.log_probs)
    values = torch.cat(self.values)
    aprobs = torch.cat(self.aprobs)

    self.batch_rewards.append(rewards)
    self.batch_log_probs.append(log_probs)
    self.batch_values.append(values)
    self.batch_aprobs.append(aprobs)

  def append_episode_score(self, s):

    self.episode_scores.append(s)

  def optimize(self):

    value_scale = 0.5
    entropy_scale = 0.0

    rewards = torch.cat(self.batch_rewards)
    log_probs = torch.cat(self.batch_log_probs)
    values = torch.cat(self.batch_values)
    aprobs = torch.cat(self.batch_aprobs)

    neg_log_prob = log_probs.mul(-1)
    policy_grad_loss = neg_log_prob * (rewards - values)
    value_loss = value_scale * F.smooth_l1_loss(values, rewards)

    loss = policy_grad_loss.mean() + value_loss

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy.parameters():
      param.grad.data.clamp_(-40, 40)
    self.optimizer.step()

    return loss.detach().cpu().numpy()

  def save_model(self, step, dest):

    model_savefile = '{0}/doom-agent-{1}.pth'.format(dest, step)
    logger.debug("Saving Doom Agent to {}".format(model_savefile))

    torch.save(self.policy.state_dict(), model_savefile)
