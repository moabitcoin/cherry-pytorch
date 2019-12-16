import time
import random
import itertools as it
from pathlib import Path
from collections import namedtuple

import gym
import numpy as np
import torch.nn.functional as F
from gym.wrappers import AtariPreprocessing


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

      env = gym.make(self.env_name)
      # default to [84, 84, 1] and in [0, 255]
      self.game = AtariPreprocessing(env)

      self.action_size = self.game.action_space.n
      self.actions = range(self.action_size)

      logger.debug('{}: Action space size {}'.format(self.env_name,
                                                     self.action_size))

      logger.info('{}: Environment setup'.format(self.env_name))

    except Exception as err:

      logger.error('{}: Error setting up env, {}'.format(self.env_name, err))