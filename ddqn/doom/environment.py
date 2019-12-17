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

    logger.info('Loading game config from {}'.format(config_file))
    logger.info('Loading scenario config from {}'.format(scenario_file))

    assert config_file.is_file(), \
        "{} no such file".format(config_file)
    assert scenario_file.is_file(), \
        "{} no such file".format(scenario_file)

    self.game.load_config(config_file.as_posix())
    self.game.set_doom_scenario_path(scenario_file.as_posix())
    self.game.set_screen_resolution(ScreenResolution.RES_320X240)
    self.game.set_screen_format(ScreenFormat.GRAY8)
    self.game.set_mode(Mode.PLAYER)

    self.game.init()

    self.action_size = self.game.get_available_buttons_size()
    self.actions = [a.tolist() for a in np.eye(self.action_size, dtype=bool)]

    logger.debug('Action space size {}'.format(self.action_size))

    logger.info('Environment setup')
