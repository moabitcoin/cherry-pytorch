import time
import random
import itertools as it
from pathlib import Path
from collections import namedtuple

import gym
import numpy as np
import torch.nn.functional as F
from baselines.common.atari_wrappers import make_atari, wrap_deepmind


from utils.helpers import get_logger

logger = get_logger(__file__)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class AtariEnvironment():

  def __init__(self, cfgs, play=False):

    self.env = None
    self.env_name = cfgs.get('name')

    assert self.env_name is not None, 'env:name not found in config'

    try:

      env = make_atari(self.env_name)
      self.env = wrap_deepmind(env)

      self.action_size = self.env.action_space.n
      self.actions = range(self.action_size)

      logger.debug('{}: Action space size {}'.format(self.env_name,
                                                     self.action_size))

      logger.info('{}: Environment setup'.format(self.env_name))

    except Exception as err:

      logger.error('{}: Error setting up env, {}'.format(self.env_name, err))

  def reset(self):

    state = self.env.reset()
    return state

  def step(self, action):

    state, reward, done, info = self.env.step(action)
    terminal = info['ale.lives'] == 0

    return state, reward, terminal, info

  def close(self):

    self.env.close()

  def update_env(self, update_fn, **kwargs):

    self.env = update_fn(self.env, **kwargs)
