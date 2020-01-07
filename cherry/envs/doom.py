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


class DoomEnvironment():

  def __init__(self, cfgs):

    scenario_name = cfgs['name']
    filepath = Path(__file__).parent

    config_file = filepath.joinpath('assets/{}.cfg'.format(scenario_name))
    scenario_file = filepath.joinpath('assets/{}.wad'.format(scenario_name))

    logger.info('Loading game config from {}'.format(config_file.name))
    logger.info('Loading scenario config from {}'.format(scenario_file.name))

    assert config_file.is_file(), \
        "{} no such file".format(config_file)
    assert scenario_file.is_file(), \
        "{} no such file".format(scenario_file)

    self.game = DoomGame()

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

    # Makes episodes start after 10 tics (~after raising the weapon)
    self.game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    self.game.set_window_visible(True)

    # Turns on the sound. (turned off by default)
    self.game.set_sound_enabled(True)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR,
    # PLAYER mode is default)
    self.game.set_mode(Mode.PLAYER)

    self.game.init()

    self.action_size = self.game.get_available_buttons_size()
    self.actions = [a.tolist() for a in np.eye(self.action_size, dtype=bool)]

    logger.debug('Action space size {}'.format(self.action_size))

    logger.info('Environment setup')

  def step(self, action):

    reward = self.game.make_action(self.actions[action])
    done = self.game.is_episode_finished()
    next_state = self.get_frame()

    return next_state, reward, done, {}

  def reset(self):

    self.game.new_episode()

    return self.get_frame()

  def get_total_reward(self):

    return self.game.get_total_reward()

  def close(self):

    self.game.close()

  def get_frame(self):

    state = self.game.get_state()

    return state.screen_buffer if state is not None else None

  def update_env(self, update_fn, **kwargs):

    pass
