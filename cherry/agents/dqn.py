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
from gym import wrappers
import torch.optim as optim
import torch.nn.functional as F
from skvideo.io import FFmpegWriter as vid_writer
from torchvision.transforms import Compose, CenterCrop, Resize, ToPILImage

from cherry.agents import ReplayBuffer
from utils.helpers import get_logger, write_model, OPTS


class DQN():

  def __init__(self, cfgs, model=None, model_file=None,
               device=None, log_level='info'):

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
    self.state_len = cfgs['state_len']
    self.action_size = cfgs['action_size']
    self.input_transforms = cfgs['input_transforms']
    self.grad_clip = cfgs['grad_clip']
    self.device = device
    self.eps = self.max_eps
    self.state_size = [self.state_len] + self.input_shape

    assert self.input_shape is not None, 'Input shape has to be not None'
    assert self.action_size is not None, 'Action size has to non None'
    assert self.device is not None, 'Device has to be CPU/GPU'

    self.zero_state = torch.zeros([1] + self.input_shape, dtype=torch.uint8)

    self.logger = get_logger(__file__, log_level=log_level)

    self.transform = self.state_transformer()

    self.policy = model(self.state_size, self.action_size,
                        self.device).to(self.device)

    # self.policy.apply(self.policy.init_weights)

    self.target = model(self.state_size, self.action_size,
                        self.device).to(self.device)

    self.target.load_state_dict(self.policy.state_dict())
    self.target.eval()

    optimizer = OPTS.get(cfgs['opt_name'])

    self.optimizer = optimizer(self.policy.parameters(), lr=self.lr)

    self.reset()
    buffer_shape = list(self.get_state(complete=True).shape)[1:]

    self.replay = ReplayBuffer(self.replay_size, buffer_shape,
                               1, device=self.device)
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

    no_history = [self.zero_state for _ in range(self.state_len + 1)]
    self.history = deque(no_history, maxlen=self.state_len + 1)

  def load_model(self, model_file):

    self.logger.info('Loading agent weights from {}'.format(model_file))
    self.policy.load_state_dict(torch.load(model_file))

  def eval(self):

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

    state_batch = states[:, :self.state_len]
    next_state_batch = states[:, 1:]

    q_values, _ = self.policy(state_batch)
    q_values = q_values.gather(1, action)
    q_values_next, _ = self.target(next_state_batch)
    q_values_next = q_values_next.max(1)[0].detach()

    # Bellman Equation : Computes the expected Q values (target)
    q_values_target = (q_values_next * self.gamma) * (1. - done[:, 0]) + reward[:, 0]

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_values, q_values_target.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_value_(self.policy.parameters(), self.grad_clip)
    self.optimizer.step()

  def update_target(self, step):

    self.logger.debug('Updating agent at {}'.format(step))
    self.target.load_state_dict(self.policy.state_dict())

  def show_score(self, pbar, step):

    total_score = np.sum(self.scores, initial=0.0)

    pbar.set_description('Reward : {0:.3f}, Eps : {1:.4f}, '
                         'Buffer : {2}'.format(total_score, self.eps,
                                               len(self.replay)))

    self.top_scr = total_score if self.top_scr < total_score else self.top_scr

  def train(self, env, train_cfgs, gitsha, model_dest, render):

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
          ep_reward = self.get_episode_rewards()

          self.reset()
          next_frame = env.reset()

          self.append_state(next_frame)

          train_step.set_description('{0}/{1}, Reward : {2:.3f}, '
                                     'Eps : {3:.4f}'.format(ep, step,
                                                            ep_reward,
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

  def play(self, env, test_cfgs, gitsha, render):

    self.eval()

    state_dest = test_cfgs['state_dest']
    test_episodes = test_cfgs['n_test_episodes']
    max_steps = test_cfgs['max_steps']

    vid_dst = Path(state_dest)
    vid_dst.mkdir(parents=True, exist_ok=True)

    env.update_env(wrappers.Monitor, directory=vid_dst.as_posix(), force=True)
    test_ep = tqdm.tqdm(range(test_episodes), ascii=True, unit='episode')

    for ep in test_ep:

      vid_file = vid_dst.joinpath('episode-{1:03d}-{0}.mp4'.format(gitsha, ep))

      writer = vid_writer(vid_file.as_posix(), outputdict={'-vcodec': 'h264',
                                                           '-b': '300000000'})

      self.reset()
      # no exploration
      self.eps = 0.0

      state = env.reset()

      self.append_state(state)
      writer.writeFrame(state)

      test_step = tqdm.tqdm(range(max_steps), ascii=True, unit='stp')

      for step in test_step:

        state = self.get_state()
        action = self.get_action(state)
        next_state, reward, done, info = env.step(action)
        self.append_reward(reward)

        if done:
          next_state = env.reset()
          reward = self.get_episode_rewards()

          self.reset()
          test_step.set_description('{0}/{1} Reward : {2:.3f}'.format(ep, step,
                                                                      reward))
        self.append_state(next_state)
        writer.writeFrame(next_state)

      writer.close()
