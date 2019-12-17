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

  def __init__(self, cfgs):

    self.game = None
    self.env_name = cfgs.get('env_name')

    assert self.env_name is not None, 'env_name not found in config'

    try:

      env = make_atari(self.env_name)
      self.game = wrap_deepmind(env)

      self.action_size = self.game.action_space.n
      self.actions = range(self.action_size)

      logger.debug('{}: Action space size {}'.format(self.env_name,
                                                     self.action_size))

      logger.info('{}: Environment setup'.format(self.env_name))

    except Exception as err:

      logger.error('{}: Error setting up env, {}'.format(self.env_name, err))

  def reset(self):

    state = self.game.reset()

    # 84x84x1 -> 1x84x84
    return state.transpose((2, 0, 1))

  def step(self, action):

    state, reward, done, info = self.game.step(self.actions[action])

    # 84x84x1 -> 1x84x84
    state = state.transpose((2, 0, 1))

    return state, reward, done, info
