# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
import os
import sys
import math
from collections import deque, namedtuple

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, Grayscale, Resize

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.memory = []
    self.position = 0

  def push(self, *args):
    """Saves a transition."""
    if len(self.memory) < self.capacity:
      self.memory.append(None)
    self.memory[self.position] = Transition(*args)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class DoomNet(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, lr):

    self.input_shape = input_shape
    self.state_size = state_size
    self.action_size = action_size
    self.lr = lr

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

    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))

    return self.head(x.view(x.size(0), -1))


class AgentOfDoom():

  def __init__(self, cfgs, device=None):

    self.crop_shape = cfgs['crop_shape']
    self.input_shape = cfgs['input_shape']
    self.learning_rate = cfgs['learning_rate']
    self.gamma = cfgs['gamma']
    self.max_eps = cfgs['max_eps']
    self.min_eps = cfgs['min_eps']
    self.eps_decay = cfgs['eps_decay']
    self.replay_size = cfgs['replay_size']
    self.action_size = cfgs['action_size']
    self.state_size = cfgs['state_size']
    self.device = device
    self.history = None

    assert self.device is not None, "Device has to be CPU/GPU"

    self.transform = Compose([Grayscale(), CenterCrop(self.crop_shape),
                              Resize(self.input_shape)])

    self.policy = DoomNet(self.input_shape, self.state_size, self.action_size)
    self.target = DoomNet(self.input_shape, self.state_size, self.action_size)

    self.target.load_state_dict(self.policy.state_dict())
    self.target.eval()

    self.optimizer = optim.RMSprop(policy_net.parameters())
    self.replay = ReplayBuffer(self.replay_size)

  def restart(self):

    no_history = [np.zeros(self.input_shape, dtype=np.int)
                  for i in range(self.state_size)]
    self.history = deque(no_history, maxlen=self.state_size)

  def get_action(self, state):

    if random.random() > self.eps:
      state = torch.from_numpy(state)
      with torch.no_grad():
        return self.policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(self.action_size)]],
                            device=self.device, dtype=torch.long)

  def set_eps(self, step):
    self.eps = self.min_eps + (self.max_eps - self.min_eps) * \
            math.exp(-1. * step / self.eps_decay)

  def set_history(self, state, new_episode=False):

    state = self.transform(state)

    if new_episode:
      _ = [self.history.append(frame) for _ in range(self.state_size)]
    else:
      self.history.append(frame)

    return stacked_state

  def get_history(self):
    return np.stack(self.history, axis=2)

  def push_to_memory(self, state, action, next_state, reward):

    self.memory.push([state, action, next_state, reward])

  def update(self, batch_size):

    if len(self.memory) < batch_size:
      return
