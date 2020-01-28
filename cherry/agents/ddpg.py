# https://spinningup.openai.com/en/latest/algorithms/ddpg.html#pseudocode
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


class DDPG():

  def __init__(self, cfgs, model=None, model_file=None,
               device=None, log_level='info'):

    self.history = None
    self.losses = None
    self.rewards = None
    self.zero_state = None
    self.ep_rewards = None
    self.rr = 10.0
    self.actor_lr = cfgs['actor_lr']
    self.critic_lr = cfgs['critic_lr']
    self.gamma = cfgs['gamma']
    self.tau = cfgs['tau']
    self.replay_size = cfgs['replay_size']
    self.state_len = cfgs['state_len']
    self.input_shape = cfgs['input_shape']
    self.crop_shape = cfgs.get('crop_shape')
    self.action_size = cfgs['action_size']
    self.input_transforms = cfgs['input_transforms']
    self.grad_clip = cfgs['grad_clip']
    self.init_weights = cfgs.get('init_weights')
    self.continous = cfgs.get('continous')
    self.grad_clip = cfgs.get('grad_clip')
    self.device = device
    self.state_size = [self.state_len] + self.input_shape

    assert self.input_shape is not None, 'Input shape has to be not None'
    assert self.action_size is not None, 'Action size has to non None'
    assert self.device is not None, 'Device has to be CPU/GPU'

    self.zero_state = torch.zeros(self.state_size, dtype=torch.uint8)

    self.logger = get_logger(__file__, log_level=log_level)

    self.transform = self.state_transformer()

    self.actor = model(self.state_size, self.action_size,
                       self.device).to(self.device)

    self.critic = model(self.state_size, self.action_size,
                        self.device, continous=self.continous).to(self.device)

    if self.init_weights:
      self.actor.apply(self.actor.init_weights)
      self.critic.apply(self.critic.init_weights)

    self.actor_target = model(self.state_size, self.action_size,
                              self.device).to(self.device)

    self.critic_target = model(self.state_size, self.action_size,
                               self.device, continous=self.continous).to(self.device)

    if model_file:
      self.load_model(model_file)

    self.update_target(0)

    optimizer = OPTS.get(cfgs['opt_name'])

    self.actor_optimizer = optimizer(self.actor.parameters(),
                                     lr=self.actor_lr)
    self.critic_optimizer = optimizer(self.critic.parameters(),
                                      lr=self.critic_lr)

    self.reset()
    buffer_shape = list(self.get_state(complete=True).shape)[1:]

    self.replay = ReplayBuffer(self.replay_size, buffer_shape,
                               self.action_size, state_type=torch.float32,
                               action_type=torch.float32, device=self.device)

  def state_transformer(self):

    if not self.input_transforms:
      return None

    transforms = [ToPILImage()]

    if 'crop' in self.input_transforms:
      transforms.append(CenterCrop(self.crop_shape))
    if 'resize' in self.input_transforms:
      transforms.append(Resize(self.input_shape))

    return Compose(transforms)

  def flash_episode(self):

    self.rewards = []

  def reset(self):

    self.ep_rewards = []

    no_history = [self.zero_state for _ in range(self.state_len + 1)]
    self.history = deque(no_history, maxlen=self.state_len + 1)

    self.flash_episode()

  def load_model(self, model_file):

    self.logger.info('Loading agent weights from {}'.format(model_file))
    self.actor.load_state_dict(torch.load(model_file))

  def eval(self):

    self.actor.eval()

  def scale_action(self, q):

    # return 0.5 * (self.env_hi + self.env_lo) * (F.tanh(q) + 1) - self.env_lo
    return self.env_hi * torch.tanh(q)

  def get_action(self, state):

    with torch.no_grad():
      q, _ = self.actor(state)

    return self.scale_action(q)[0].detach().cpu().numpy()

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

    action = torch.Tensor(action)

    self.replay.push(states, action, reward, done)

  def get_episode_rewards(self):

    return np.sum(self.rewards)

  def append_episode_reward(self, reward):

    self.rr = 0.05 * reward + (1 - 0.05) * self.rr

    self.ep_rewards.append(self.rr)

  def optimize(self, batch_size=32):

    if len(self.replay) < batch_size:
      return

    batch = self.replay.sample(batch_size)

    states, action, reward, done = batch

    state_batch = states[:, :self.state_len]
    next_state_batch = states[:, 1:]

    # Optimize the critic model
    next_action_batch, _ = self.actor_target(next_state_batch)
    next_action_batch = self.scale_action(next_action_batch)
    _, q_values_next = self.critic_target(next_state_batch, next_action_batch)

    # Bellman Equation : Computes the expected Q values (target)
    q_values_target = (q_values_next * self.gamma) * (1. - done) + reward

    _, q_values = self.critic(state_batch, action)

    # critic loss
    critic_loss = F.smooth_l1_loss(q_values, q_values_target)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    if self.grad_clip:
      nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
    self.critic_optimizer.step()

    # Optimize actor network
    action, _ = self.actor(state_batch)
    action = self.scale_action(action)
    _, q_values = self.critic(state_batch, action)

    actor_loss = -q_values.mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    if self.grad_clip:
      nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
    self.actor_optimizer.step()

  def update_target(self, step):

    # Update the frozen target models
    self.copy_model(self.critic, self.critic_target, self.tau)
    self.copy_model(self.actor, self.actor_target, self.tau)

    self.logger.debug('Updating agent at {}'.format(step))

  def copy_model(self, src, dest, w):

    for src_param, dest_param in zip(src.parameters(), dest.parameters()):
      dest_param.data.copy_(w * src_param.data + (1 - w) * dest_param.data)

  def set_action_limits(self, limits):

    self.env_lo = torch.Tensor(limits[0]).to(self.device)
    self.env_hi = torch.Tensor(limits[1]).to(self.device)

  def train(self, env, train_cfgs, gitsha, model_dest):

    batch_size = train_cfgs['batch_size']
    update_target = train_cfgs['update_target']
    save_model = train_cfgs['save_model']
    train_eps = train_cfgs['n_train_episodes']
    max_steps = train_cfgs['max_steps']
    policy_update = train_cfgs['policy_update']
    n_exploration_steps = train_cfgs['n_exploration_steps']

    train_ep = tqdm.tqdm(range(train_eps), ascii=True,
                         unit='episode', leave=False)

    self.set_action_limits(env.action_limits())

    global_step = 0

    for ep in train_ep:

      self.reset()
      frame = env.reset()

      self.append_state(frame)

      train_step = tqdm.tqdm(range(max_steps), ascii=True,
                             unit='stp', leave=False)

      done = False

      for step in train_step:

        global_step = ep * max_steps + step

        state = self.get_state()
        action = self.get_action(state) if global_step > n_exploration_steps \
            else env.sample()
        next_state, reward, done, info = env.step(action)
        self.append_reward(reward)

        if done:
          ep_reward = self.get_episode_rewards()
          self.append_episode_reward(ep_reward)

          self.flash_episode()
          next_state = env.reset()

        self.append_state(next_state)

        states = self.get_state(complete=True)
        self.push_to_memory(states, action, reward, done)

        if global_step % policy_update == 0:
          self.optimize(batch_size=batch_size)

        if global_step % update_target == 0:
          self.update_target(global_step)

        if global_step % save_model == 0:
          tag = '{0:09d}-{1}'.format(global_step, gitsha)
          write_model(self.actor, tag, model_dest)

      if not done:
        ep_reward = self.get_episode_rewards()
        self.append_episode_reward(ep_reward)

      mean_reward = np.mean(self.ep_rewards)
      train_ep.set_description('Average reward: {:.3f}'.format(mean_reward))

      best_reward = np.max(self.ep_rewards)
      if best_reward >= env.env_solution:
        self.logger.info('Solved! At epside {}'
                         ' reward {:.3f} > {:.3f}'.format(ep, best_reward,
                                                          env.env_solution))
        break

    tag = 'final-{0}'.format(gitsha)
    write_model(self.actor, tag, model_dest)

  def play(self, env, test_cfgs, gitsha):

    self.eval()

    state_dest = test_cfgs['state_dest']
    test_episodes = test_cfgs['n_test_episodes']
    max_steps = test_cfgs['max_steps']

    vid_dst = Path(state_dest)
    vid_dst.mkdir(parents=True, exist_ok=True)

    env.update_env(wrappers.Monitor, directory=vid_dst.as_posix(), force=True)
    test_ep = tqdm.tqdm(range(test_episodes), ascii=True,
                        unit='episode', leave=False)

    self.set_action_limits(env.action_limits())

    for ep in test_ep:

      vid_file = vid_dst.joinpath('episode-{1:03d}-{0}.mp4'.format(gitsha, ep))

      self.reset()
      state = env.reset()

      self.append_state(state)

      test_step = tqdm.tqdm(range(max_steps), ascii=True,
                            unit='stp', leave=False)

      for step in test_step:

        state = self.get_state()
        action = self.get_action(state)
        next_state, reward, done, info = env.step(action)
        self.append_reward(reward)

        if done:
          ep_reward = self.get_episode_rewards()
          test_ep.set_description('{0} Reward : {1:.3f}'.format(ep, ep_reward))

          self.reset()
          next_state = env.reset()

        self.append_state(next_state)

    env.close()
