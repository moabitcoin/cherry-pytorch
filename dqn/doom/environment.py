import time
import random
from pathlib import Path
from collections import namedtuple

import numpy as np
from vizdoom import DoomGame, ScreenResolution, ScreenFormat

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DoomEnvironment():

  def __init__(self, cfgs):

    self.game = DoomGame()

    config_file = Path(cfgs['game_config'])
    scenario_file = Path(cfgs['scenario_config'])

    assert config_file.is_file(), \
        "{} no such file".format(config_file)
    assert scenario_file.is_file(), \
        "{} no such file".format(scenario_file)

    # self.game.load_config(config_file.as_posix())
    self.game.set_doom_scenario_path(scenario_file.as_posix())
    self.game.set_screen_resolution(ScreenResolution.RES_320X240)
    self.game.set_screen_format(ScreenFormat.BGR24)
    self.game.init()

    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    self.actions = [left, right, shoot]

    self.action_size = len(self.actions)
