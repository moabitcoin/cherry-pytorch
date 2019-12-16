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


class AtariNet(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, lr):

    super(AtariNet, self).__init__()

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


class AgentOfAtari():

  def __init__(self, cfgs, action_size=None, device=None, model_file=None):

    self.history = None
    self.losses = None
    self.rewards = None
    self.null_state = None
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

    assert self.device is not None, "Device has to be CPU/GPU"

    transforms = [ToPILImage(), CenterCrop(self.crop_shape)] \
        if self.crop_shape else []

    transforms.append(ToTensor())

    self.transform = Compose(transforms)

    self.policy = AtariNet(self.input_shape, self.state_size,
                           self.action_size, self.lr).to(self.device)
    self.target = AtariNet(self.input_shape, self.state_size,
                           self.action_size, self.lr).to(self.device)

    self.target.load_state_dict(self.policy.state_dict())
    self.target.eval()

    self.optimizer = optim.RMSprop(self.policy.parameters(), lr=self.lr)
    self.replay = ReplayBuffer(self.replay_size)

    if model_file:
      self.load_model(model_file)

  def flash(self):

    self.losses = []
    self.scores = []

  def reset(self):

    self.top_scr = 0.0

    no_history = [self.null_state for _ in range(self.state_size)]
    self.history = deque(no_history, maxlen=self.state_size)

    self.flash()

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
    self.eps -= (self.max_eps - self.min_eps) / self.eps_decay
    self.eps = max(self.eps, self.min_eps)

  def set_history(self, frame, new_episode=False):

    replicas = self.state_size if new_episode else 1

    self.history += [self.transform(frame).to(self.device)
                     for _ in range(replicas)]

  def get_history(self, done=False):

    return self.null_state if done else \
      torch.cat([h for h in self.history]).unsqueeze(0)

  def push_to_memory(self, state, action, next_state, reward):

    mem = [state, action, next_state]
    mem = [m.cpu().detach().numpy()
           if m is not self.null_state else self.null_state for m in mem]
    (state, action, next_state) = mem

    reward = np.array([reward], dtype=np.float32)
    self.replay.push(state, action, next_state, reward)

  def update_scores(self, score):

    self.scores.append(score)

  def update(self, batch_size=32):

    if len(self.replay) < batch_size:
      return

    transitions = self.replay.sample(batch_size)
    # [(a, b), (c, d), (e, f)] -> [(a, c, e), (b, d, f)]
    batch = Transition(*zip(*transitions))

    def tensorize_inputs(inputs):

      input_tensors = [None if a is None else
                       torch.tensor(a).to(self.device) for a in inputs]

      return input_tensors

    memory = [batch.state, batch.action, batch.next_state, batch.reward]
    memory = list(map(tensorize_inputs, memory))
    batch = Transition(*memory)

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

  def update_target(self, step):

    logger.debug('Updating agent at {}'.format(step))
    self.target.load_state_dict(self.policy.state_dict())

  def save_model(self, step, dest):

    model_savefile = '{0}/atari-agent-{1}.pth'.format(dest, step)
    logger.debug("Saving Atari Agent to {}".format(model_savefile))

    torch.save(self.target.state_dict(), model_savefile)

  def show_score(self, pbar, step):

    total_score = np.sum(self.scores, initial=0.0)
    mean_loss = 0.0 if self.losses == [] else np.mean(self.losses)

    pbar.set_description('Step : {0} Reward : {1:.3f}, Loss : {2:.4f}, Eps'
                         ' : {3:.4f}, Buffer : {4}'.format(step,
                                                           total_score,
                                                           mean_loss,
                                                           self.eps,
                                                           len(self.replay)))

    self.top_scr = total_score if self.top_scr < total_score else self.top_scr
