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

    super(DoomNet, self).__init__()

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

  def __init__(self, cfgs, action_size=None, device=None, model_file=None):

    self.history = None
    self.losses = None
    self.rewards = None
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

    end_state = np.zeros([1] + self.input_shape, np.float32)
    end_state = np.ascontiguousarray(end_state)
    self.end_state = torch.tensor(end_state, device=self.device)

    assert self.device is not None, "Device has to be CPU/GPU"

    transforms = [ToPILImage()]
    if self.crop_shape:
      transforms.append(CenterCrop(self.crop_shape))
    transforms.append(Resize(self.input_shape))
    transforms.append(ToTensor())

    self.transform = Compose(transforms)

    self.policy = DoomNet(self.input_shape, self.state_size,
                          self.action_size, self.lr).to(self.device)
    self.target = DoomNet(self.input_shape, self.state_size,
                          self.action_size, self.lr).to(self.device)

    self.target.load_state_dict(self.policy.state_dict())
    self.target.eval()

    self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.lr)
    self.replay = ReplayBuffer(self.replay_size)

    if model_file:
      self.load_model(model_file)

  def restart(self):

    no_history = [self.end_state for i in range(self.state_size)]
    self.history = deque(no_history, maxlen=self.state_size)

  def reset(self):

    self.losses = []
    self.scores = []

    self.restart()

  def load_model(self, model_file):

    logger.info('Loading agent weights from {}'.format(model_file))

    self.policy.load_state_dict(torch.load(model_file))
    self.policy.eval()

  def get_action(self, state):

    if random.random() > self.eps:
      with torch.no_grad():
        return self.policy(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(self.action_size)]],
                            device=self.device, dtype=torch.long)

  def set_eps(self, step):
    self.eps = self.min_eps + (self.max_eps - self.min_eps) * \
            math.exp(-1. * step / self.eps_decay)

  def set_history(self, frame, new_episode=False):

    if frame is None:
      state = self.end_state
    else:
      state = self.transform(frame).to(self.device)

    history_update = [state for _ in range(self.state_size)] \
        if new_episode else [state]

    self.history += history_update

  def get_history(self):

    history = [h for h in self.history]
    history = torch.cat(history).unsqueeze(0).to(self.device)

    return history

  def push_to_memory(self, state, action, next_state, reward):

    reward = torch.tensor([reward], device=self.device)
    self.replay.push(state, action, next_state, reward)

  def update_scores(self, score):

    self.scores.append(score)

  def update(self, batch_size=32):

    if len(self.replay) < batch_size:
      return

    transitions = self.replay.sample(batch_size)
    # [(a, b), (c, d), (e, f)] -> [(a, c, e), (b, d, f)]
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)),
                                  device=self.device, dtype=torch.bool)

    non_final_states = torch.cat([s for s in batch.next_state
                                  if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    q_values = self.policy(state_batch).gather(1, action_batch)

    q_values_next = torch.zeros(batch_size, device=self.device)
    target_q_values = self.target(non_final_states).max(1)[0].detach()
    q_values_next[non_final_mask] = target_q_values

    # Compute the expected Q values (target)
    q_values_target = (q_values_next * self.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_values, q_values_target.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy.parameters():
      param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    self.losses.append(loss.item())

  def update_target(self, ep):

    logger.debug('Updating agent at {}'.format(ep))
    self.target.load_state_dict(self.policy.state_dict())

  def save_model(self, ep, dest):

    model_savefile = '{0}/doom-agent-{1}.pth'.format(dest, ep)
    logger.debug("Saving Doom Agent to {}".format(model_savefile))

    torch.save(self.target.state_dict(), model_savefile)
