# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
import os
import sys
import math
from pathlib import Path
from collections import deque, namedtuple

import tqdm
import torch
import random
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import Compose, CenterCrop, Resize, ToPILImage

from cherry.agents import ReplayBuffer
from utils.helpers import get_logger, write_model

logger = get_logger(__file__)


class DQN():

  def __init__(self, cfgs, model=None, model_file=None, device=None):

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
    self.action_size = cfgs['action_size']
    self.input_transforms = cfgs['input_transforms']
    self.device = device
    self.eps = self.max_eps

    assert self.input_shape is not None, 'Input shape has to be not None'
    assert self.action_size is not None, 'Action size has to non None'
    assert self.device is not None, 'Device has to be CPU/GPU'

    self.zero_state = torch.zeros([1] + self.input_shape, dtype=torch.uint8)

    self.transform = self.state_transformer()

    self.policy = model(self.input_shape, self.state_size,
                        self.action_size, self.device).to(self.device)

    self.policy.apply(self.policy.init_weights)

    self.target = model(self.input_shape, self.state_size,
                        self.action_size, self.device).to(self.device)

    self.target.load_state_dict(self.policy.state_dict())
    self.target.eval()

    self.optimizer = optim.Adam(self.policy.parameters(),
                                lr=self.lr, eps=1.5e-4)
    self.replay = ReplayBuffer(self.replay_size,
                               [self.state_size] + self.input_shape,
                               self.device)
    if model_file:
      self.load_model(model_file)

  def state_transformer(self):

    transforms = [ToPILImage()]

    if 'crop' in self.input_transforms:
      transforms.append(CenterCrop(self.crop_shape))
    if 'resize' in self.input_transforms:
      transforms.append(Resize(self.input_shape))

    return Compose(transforms)

  def flush_episode(self):

    self.losses = []
    self.rewards = []

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
        q, _ = self.policy(state)
        a = q.max(1)[1].cpu().view(1, 1)
    else:
      a = torch.tensor([[random.randrange(self.action_size)]],
                       device='cpu', dtype=torch.long)

    return a.numpy()[0, 0].item()

  def set_eps(self, step):

    self.eps -= (self.max_eps - self.min_eps) / self.eps_decay
    self.eps = max(self.eps, self.min_eps)

  def append_state(self, state):

    state = self.zero_state if state is None else state

    if self.transform:
      state = np.array(self.transform(state), dtype=np.uint8)
      state = np.expand_dims(state, 0)

    state = torch.from_numpy(state)

    self.history.append(state)

  def append_reward(self, r):

    self.rewards.append(r)

  def get_state(self, complete=False):

    size = [1, 0][complete]
    return torch.cat(list(self.history)[size:]).unsqueeze(0)

  def push_to_memory(self, states, action, reward, done):

    self.replay.push(states, action, reward, done)

  def get_episode_rewards(self):

    return np.sum(self.rewards)

  def optimize(self, batch_size=32):

    if len(self.replay) < batch_size:
      return

    batch = self.replay.sample(batch_size)

    states, action, reward, done = batch

    state_batch = states[:, :self.state_size]
    next_state_batch = states[:, 1:]

    q_values, _ = self.policy(state_batch)
    q_values = q_values.gather(1, action)
    q_values_next, _ = self.target(next_state_batch)
    q_values_next = q_values_next.max(1)[0].detach()

    # Compute the expected Q values (target)
    q_values_target = (q_values_next * self.gamma) * (1. - done[:, 0]) + reward[:, 0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_values, q_values_target.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(self.policy.parameters(), 1)
    self.optimizer.step()

  def update_target(self, step):

    logger.debug('Updating agent at {}'.format(step))
    self.target.load_state_dict(self.policy.state_dict())

  def show_score(self, pbar, step):

    total_score = np.sum(self.scores, initial=0.0)

    pbar.set_description('Reward : {0:.3f}, Eps : {1:.4f}, '
                         'Buffer : {2}'.format(total_score, self.eps,
                                               len(self.replay)))

    self.top_scr = total_score if self.top_scr < total_score else self.top_scr

  def train(self, env, train_cfgs, gitsha, model_dest):

    batch_size = train_cfgs['batch_size']
    update_target = train_cfgs['update_target']
    save_model = train_cfgs['save_model']
    train_eps = train_cfgs['n_train_episodes']
    max_steps = train_cfgs['max_steps']
    policy_update = train_cfgs['policy_update']

    train_ep = tqdm.tqdm(range(train_eps), ascii=True,
                         unit='episode', leave=False)

    global_step = 0

    for ep in train_ep:

      self.reset()
      frame = env.reset()

      self.append_state(frame)

      train_step = tqdm.tqdm(range(max_steps), ascii=True,
                             unit='stp', leave=False)

      for step in train_step:

        global_step = ep * max_steps + step
        self.set_eps(global_step)

        state = self.get_state()
        action = self.get_action(state)
        next_state, reward, done, info = env.step(action)
        self.append_reward(reward)
        self.append_state(next_state)

        states = self.get_state(complete=True)
        self.push_to_memory(states, action, reward, done)

        if done:
          total_score = self.get_episode_rewards()

          self.reset()
          next_frame = env.reset()

          self.append_state(next_frame)

          train_step.set_description('{0}/{1}, Reward : {2:.3f}, '
                                     'Eps : {3:.4f}'.format(ep, step,
                                                            total_score,
                                                            self.eps))

        if global_step % policy_update == 0:
          self.optimize(batch_size=batch_size)

        if global_step % update_target == 0:
          self.update_target(global_step)

        if global_step % save_model == 0:
          tag = '{0:09d}-{1}'.format(global_step, gitsha)
          write_model(self.policy, tag, model_dest)

    tag = 'final-{0}'.format(gitsha)
    write_model(self.policy, tag, model_dest)
