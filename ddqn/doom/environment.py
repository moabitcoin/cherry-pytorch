import time
import random
import itertools as it
from pathlib import Path
from collections import namedtuple

import numpy as np
import torch.nn.functional as F
from vizdoom import DoomGame, ScreenResolution, \
    ScreenFormat, GameVariable, Mode, Button

from utils.helpers import get_logger

logger = get_logger(__file__)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DoomEnvironment():

  def __init__(self, cfgs):

    self.game = DoomGame()

    config_file = Path(cfgs['game_config'])
    scenario_file = Path(cfgs['scenario_config'])

    assert config_file.is_file(), \
        "{} no such file".format(config_file)
    logger.info('Loading game config from {}'.format(config_file))
    self.game.load_config(config_file.as_posix())

    assert scenario_file.is_file(), \
        "{} no such file".format(scenario_file)

    logger.info('Loading scenario config from {}'.format(scenario_file))
    self.game.set_doom_scenario_path(scenario_file.as_posix())

    self.game.set_screen_resolution(ScreenResolution.RES_320X240)
    self.game.set_screen_format(ScreenFormat.GRAY8)
    self.game.set_mode(Mode.PLAYER)
    try:
      self.game.init()
    except Exception as err:
      logger.error('Error setting up doom game {}'.format(err))

    self.action_size = self.game.get_available_buttons_size()
    self.actions = [a.tolist() for a in np.eye(self.action_size, dtype=bool)]

    logger.debug('Action space size {}'.format(self.action_size))

    logger.info('Environment setup')

  def step(self, action, skip_steps=4):

    next_states = []
    reward = 0
    done = False

    for _ in range(skip_steps):

      reward += self.game.make_action(self.actions[action])
      done = self.game.is_episode_finished()

      next_state = self.get_frame()
      next_states.append(next_state)

      if done:
        break

    return next_states, reward, done

  def reset(self):

    self.game.new_episode()

  def get_total_reward(self):

    return self.game.get_total_reward()

  def close(self):

    self.game.close()

  def get_frame(self):

    state = self.game.get_state()

    return state.screen_buffer if state is not None else None
