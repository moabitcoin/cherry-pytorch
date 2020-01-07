# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#dqn-algorithm
import os
import sys
import math
from collections import deque, namedtuple

import tqdm
import torch
import random
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torchvision.transforms import Compose, CenterCrop, \
    Grayscale, Resize, ToPILImage, ToTensor

from utils.helpers import get_logger, write_model

logger = get_logger(__file__)


class VPG():

  def __init__(self, cfgs, model=None, model_file=None, device=None):

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
    self.action_size = cfgs['action_size']
    self.crop_shape = cfgs.get('crop_shape')
    self.input_shape = cfgs.get('input_shape')
    self.input_transforms = cfgs.get('input_transforms')
    self.device = device

    assert self.input_shape, 'Input shape has to be not None'
    assert self.action_size, 'Action size has to non None'
    assert self.device, 'Device has to be CPU/GPU'

    self.zero_state = torch.zeros([1] + self.input_shape, dtype=torch.uint8)

    self.transform = self.state_transformer()

    self.policy = model(self.input_shape, self.state_size,
                        self.action_size, self.device).to(self.device)

    self.value = model(self.input_shape, self.state_size,
                       self.action_size, self.device).to(self.device)

    # self.policy.apply(self.policy.init_weights)

    self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                       lr=self.policy_lr)
    self.value_optimizer = optim.Adam(self.value.parameters(),
                                      lr=self.value_lr)

    if model_file:
      self.load_model(model_file)

    logger.info('Done setting up {} Agent'.format(__class__.__name__))

  def state_transformer(self):

    if not self.input_transforms:
      return None

    transforms = [ToPILImage()]
    if self.crop_shape:
      transforms.append(CenterCrop(self.crop_shape))
    transforms.append(Resize(self.input_shape))

    return Compose(transforms)

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

    if self.transform:
      state = np.array(self.transform(state), dtype=np.uint8).expand_dims(0)

    state = torch.from_numpy(state)
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

    if ep_length == 1:
      # std is not defined for array of length 1
      return

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

  def train(self, env, train_cfgs, gitsha, model_dest):

    save_model = train_cfgs['save_model']
    train_eps = train_cfgs['n_train_episodes']
    max_steps = train_cfgs['max_steps']
    env_solved = train_cfgs['env_solution']

    train_ep = tqdm.tqdm(range(train_eps), ascii=True, unit='ep', leave=True)

    running_reward = 10

    for ep in train_ep:

      self.reset()
      state = env.reset()

      self.append_state(state)

      train_step = tqdm.tqdm(range(max_steps), ascii=True,
                             unit='stp', leave=False)

      for step in train_step:

        state = self.get_state()
        action = self.get_action(state)
        next_state, reward, done, info = env.step(action)
        self.append_reward(reward)
        self.append_state(next_state)

        if done:

          running_reward = 0.05 * self.get_episode_rewards() + \
              (1 - 0.05) * running_reward

          self.append_episode_reward(running_reward)

          self.discount_episode()
          self.flash_episode()
          state = env.reset()
          self.append_state(state)

      loss = self.optimize()

      mean_rewards = np.mean(self.ep_rewards)
      train_ep.set_description('Average reward: {:.3f}'.format(mean_rewards))

      if ep % save_model == 0:
        tag = '{0:09d}-{1}'.format(ep * max_steps, gitsha)
        write_model(self.policy, tag, model_dest)

      best_reward = np.max(self.ep_rewards)
      if best_reward >= env_solved:
        logger.info('Solved! At epside {}'
                    ' reward {:.3f} > {:.3f}'.format(ep, best_reward,
                                                     env_solved))
        break

    tag = 'final-{0}'.format(gitsha)
    write_model(self.policy, tag, model_dest)
