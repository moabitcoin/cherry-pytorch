import time
import random
from pathlib import Path
from collections import namedtuple

import numpy as np
from vizdoom import DoomGame

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DoomEnvironment():

  def __init__(self, cfgs):

    self.game = DoomGame()

    config_file = Path(cfgs['config_file'])
    scenario_file = Path(cfgs['scenario_file'])

    assert config_file.isfile(), \
        "{} no such file".format(config_file)
    assert scenario_file.isfile(), \
        "{} no such file".format(scenario_file)

    self.game.load_config(config_file.as_posix())
    self.game.set_doom_scenario_path(scenario_file.as_posix())
    self.game.init()

    # Here our possible actions
    left = [1, 0, 0]
    right = [0, 1, 0]
    shoot = [0, 0, 1]
    self.actions = [left, right, shoot]

    self.action_size = len(self.actions)
