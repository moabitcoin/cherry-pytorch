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
from torchvision.transforms import Compose, CenterCrop, \
    Grayscale, Resize, ToPILImage, ToTensor


from utils.helpers import get_logger

logger = get_logger(__file__)


class ReplayBuffer(object):

  def __init__(self, capacity, state_shape, device):

    (c, h, w) = state_shape
    self.size = 0
    self.position = 0
    self.capacity = capacity
    self.device = device
    self.states = torch.zeros((capacity, c + 1, h, w), dtype=torch.uint8)
    self.actions = torch.zeros((capacity, 1), dtype=torch.long)
    self.rewards = torch.zeros((capacity, 1), dtype=torch.int8)
    self.dones = torch.zeros((capacity, 1), dtype=torch.bool)

  def push(self, *args):
    """Saves a transition."""

    s, a, r, d = args

    self.states[self.position] = s
    self.actions[self.position, 0] = a
    self.rewards[self.position, 0] = r
    self.dones[self.position, 0] = d
    self.position = (self.position + 1) % self.capacity

    self.size = max(self.size, self.position)

  def sample(self, batch_size):

    i = torch.randint(0, high=self.size, size=(batch_size,))
    s = self.states[i]
    a = self.actions[i].to(self.device)
    r = self.rewards[i].to(self.device).float()
    d = self.dones[i].to(self.device).float()
    return s, a, r, d

  def __len__(self):
    return self.size


class AtariNet(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, lr, device):

    super(AtariNet, self).__init__()

    self.lr = lr
    self.device = device
    self.input_shape = input_shape
    self.state_size = state_size
    self.action_size = action_size

    (w, h) = self.input_shape

    self.conv1 = nn.Conv2d(self.state_size, 16, kernel_size=5, stride=2)
    self.bn1 = nn.BatchNorm2d(16)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.bn2 = nn.BatchNorm2d(32)
    self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
    self.bn3 = nn.BatchNorm2d(32)

    def feat_shape(size, kernel_size=5, stride=2):
      return (size - (kernel_size - 1) - 1) // stride + 1
    convw = feat_shape(feat_shape(feat_shape(w)))
    convh = feat_shape(feat_shape(feat_shape(h)))
    feat_spatial_shape = convw * convh * 32
    self.head = nn.Linear(feat_spatial_shape, self.action_size)

  def forward(self, x):

    x = x.to(self.device).float() / 255.
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))

    return self.head(x.view(x.size(0), -1))


class AgentOfAtari():

  def __init__(self, cfgs, action_size=None, device=None, model_file=None):

    self.history = None
    self.losses = None
    self.rewards = None
    self.zero_state = None
    self.top_scr = 0.0
    self.crop_shape = cfgs['crop_shape']
    self.input_shape = cfgs['input_shape']
    self.lr = cfgs['lr']
    self.gamma = cfgs['gamma']
    self.max_eps = cfgs['max_eps']
    self.min_eps = cfgs['min_eps']
    self.eps_decay = cfgs['eps_decay']
    self.replay_size = cfgs['replay_size']
    self.state_size = cfgs['state_size']
    self.action_size = action_size
    self.device = device
    self.eps = self.max_eps

    assert self.input_shape is not None, 'Input shape has to be not None'
    assert self.action_size is not None, 'Action size has to non None'
    assert self.device is not None, 'Device has to be CPU/GPU'

    self.zero_state = torch.zeros([1] + self.input_shape, dtype=torch.uint8)

    self.policy = AtariNet(self.input_shape, self.state_size,
                           self.action_size, self.lr,
                           self.device).to(self.device)
    self.target = AtariNet(self.input_shape, self.state_size,
                           self.action_size, self.lr,
                           self.device).to(self.device)

    self.target.load_state_dict(self.policy.state_dict())
    self.target.eval()

    self.optimizer = optim.Adam(self.policy.parameters(),
                                lr=self.lr, eps=1.5e-4)
    self.replay = ReplayBuffer(self.replay_size,
                               [self.state_size] + self.input_shape,
                               self.device)

    if model_file:
      self.load_model(model_file)

  def flush_episode(self):

    self.losses = []
    self.scores = []

  def reset(self):

    self.top_scr = 0.0
    self.flush_episode()

    no_history = [self.zero_state for _ in range(self.state_size + 1)]
    self.history = deque(no_history, maxlen=self.state_size + 1)

  def load_model(self, model_file):

    logger.info('Loading agent weights from {}'.format(model_file))

    self.policy.load_state_dict(torch.load(model_file))
    self.policy.eval()

  def get_action(self, state):

    if random.random() > self.eps:
      with torch.no_grad():
        a = self.policy(state).max(1)[1].cpu().view(1, 1)
    else:
      a = torch.tensor([[random.randrange(self.action_size)]],
                       device='cpu', dtype=torch.long)

    return a.numpy()[0, 0].item()

  def set_eps(self, step):

    self.eps -= (self.max_eps - self.min_eps) / self.eps_decay
    self.eps = max(self.eps, self.min_eps)

  def append_state(self, state):

    state = torch.from_numpy(state).view(1, self.input_shape[0],
                                         self.input_shape[1])

    self.history.append(state)

  def get_state(self, complete=False):

    size = [self.state_size, self.state_size + 1][complete]
    return torch.cat(list(self.history)[: size]).unsqueeze(0)

  def push_to_memory(self, states, action, reward, done):

    self.replay.push(states, action, reward, done)

  def update_scores(self, score):

    self.scores.append(score)

  def optimize(self, batch_size=32):

    if len(self.replay) < batch_size:
      return

    batch = self.replay.sample(batch_size)

    states, action, reward, done = batch

    state_batch = states[:, :self.state_size]
    next_state_batch = states[:, 1:]

    q_values = self.policy(state_batch).gather(1, action)
    q_values_next = self.target(next_state_batch).max(1)[0].detach()

    # Compute the expected Q values (target)
    q_values_target = (q_values_next * self.gamma) * (1. - done[:, 0]) + reward[:, 0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_values, q_values_target.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy.parameters():
      param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

  def update_target(self, step):

    logger.debug('Updating agent at {}'.format(step))
    self.target.load_state_dict(self.policy.state_dict())

  def save_model(self, step, dest):

    model_savefile = '{0}/atari-agent-{1}.pth'.format(dest, step)
    logger.debug("Saving Atari Agent to {}".format(model_savefile))

    torch.save(self.target.state_dict(), model_savefile)

  def show_score(self, pbar, step):

    total_score = np.sum(self.scores, initial=0.0)

    pbar.set_description('Reward : {0:.3f}, Eps : {1:.4f}, '
                         'Buffer : {2}'.format(total_score, self.eps,
                                               len(self.replay)))

    self.top_scr = total_score if self.top_scr < total_score else self.top_scr
