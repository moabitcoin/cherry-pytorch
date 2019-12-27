import time
import random
import itertools as it
from pathlib import Path
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn.functional as F


from utils.helpers import get_logger

logger = get_logger(__file__)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ClassicControlEnvironment():

  def __init__(self, cfgs, play=False):

    self.game = None
    self.env_name = cfgs.get('env_name')
    self.seed = cfgs.get('seed')

    assert self.env_name is not None, 'env_name not found in config'

    try:

      self.env = gym.make(self.env_name)
      self.env.seed(self.seed)

      torch.manual_seed(self.seed)

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

    next_state, reward, done, info = self.env.step(action)
    return next_state, reward, done, info
