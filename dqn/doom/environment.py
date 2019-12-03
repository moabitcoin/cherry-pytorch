import time
import random
import itertools as it
from pathlib import Path
from collections import namedtuple

import numpy as np
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

    # Enables depth buffer.
    self.game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    self.game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    self.game.set_automap_buffer_enabled(True)

    # Sets other rendering options (all of these options except
    # crosshair are enabled (set to True) by default)
    self.game.set_render_hud(True)
    self.game.set_render_minimal_hud(False)  # If hud is enabled
    self.game.set_render_crosshair(False)
    self.game.set_render_weapon(True)
    self.game.set_render_decals(False)  # Bullet holes and blood on the walls
    self.game.set_render_particles(False)
    self.game.set_render_effects_sprites(False)  # Smoke and blood
    self.game.set_render_messages(False)  # In-game messages
    self.game.set_render_corpses(False)
    # Effect upon taking damage or picking up items
    self.game.set_render_screen_flashes(True)

    # Adds buttons that will be allowed.
    self.game.add_available_button(Button.MOVE_LEFT)
    self.game.add_available_button(Button.MOVE_RIGHT)
    self.game.add_available_button(Button.ATTACK)

    # Adds game variables that will be included in state.
    self.game.add_available_game_variable(GameVariable.AMMO2)

    # Causes episodes to finish after 200 tics (actions)
    self.game.set_episode_timeout(200)

    # Makes episodes start after 10 tics (~after raising the weapon)
    self.game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    self.game.set_window_visible(True)

    # Turns on the sound. (turned off by default)
    self.game.set_sound_enabled(True)

    # Sets the livin reward (for each move) to -1
    self.game.set_living_reward(0)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR,
    # PLAYER mode is default)
    self.game.set_mode(Mode.PLAYER)

    self.game.init()

    n_buttons = self.game.get_available_buttons_size()
    self.actions = [list(a) for a in it.product([0, 1], repeat=n_buttons)]

    self.action_size = len(self.actions)
    logger.debug('Action space size {}'.format(self.action_size))

    logger.info('Environment setup')
