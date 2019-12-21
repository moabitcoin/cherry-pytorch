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

default_reward_values = {'BASE_REWARD': 0., 'DISTANCE': 0.,
                         'KILL': 5., 'DEATH': -5., 'SUICIDE': -5.,
                         'MEDIKIT': 1., 'ARMOR': 1., 'INJURED': -1.,
                         'WEAPON': 1., 'AMMO': 1., 'USE_AMMO': -0.2}

game_variables = [
    # ('KILLCOUNT', GameVariable.KILLCOUNT),
    # ('ITEMCOUNT', GameVariable.ITEMCOUNT),
    # ('SECRETCOUNT', GameVariable.SECRETCOUNT),
    ('frag_count', GameVariable.FRAGCOUNT),
    # ('DEATHCOUNT', GameVariable.DEATHCOUNT),
    ('health', GameVariable.HEALTH),
    ('armor', GameVariable.ARMOR),
    # ('DEAD', GameVariable.DEAD),
    # ('ON_GROUND', GameVariable.ON_GROUND),
    # ('ATTACK_READY', GameVariable.ATTACK_READY),
    # ('ALTATTACK_READY', GameVariable.ALTATTACK_READY),
    ('sel_weapon', GameVariable.SELECTED_WEAPON),
    ('sel_ammo', GameVariable.SELECTED_WEAPON_AMMO),
    # ('AMMO0', GameVariable.AMMO0),  # UNK
    # ('AMMO1', GameVariable.AMMO1),  # fist weapon, should always be 0
    ('bullets', GameVariable.AMMO2),  # bullets
    ('shells', GameVariable.AMMO3),  # shells
    # ('AMMO4', GameVariable.AMMO4),  # == AMMO2
    ('rockets', GameVariable.AMMO5),  # rockets
    ('cells', GameVariable.AMMO6),  # cells
    # ('AMMO7', GameVariable.AMMO7),  # == AMMO6
    # ('AMMO8', GameVariable.AMMO8),  # UNK
    # ('AMMO9', GameVariable.AMMO9),  # UNK
    # ('WEAPON0', GameVariable.WEAPON0),  # UNK
    ('fist', GameVariable.WEAPON1),  # Fist, should be 1, unless removed
    ('pistol', GameVariable.WEAPON2),  # Pistol
    ('shotgun', GameVariable.WEAPON3),  # Shotgun
    ('chaingun', GameVariable.WEAPON4),  # Chaingun
    ('rocketlauncher', GameVariable.WEAPON5),  # Rocket Launcher
    ('plasmarifle', GameVariable.WEAPON6),  # Plasma Rifle
    ('bfg9000', GameVariable.WEAPON7),  # BFG9000
    # ('WEAPON8', GameVariable.WEAPON8),  # UNK
    # ('WEAPON9', GameVariable.WEAPON9),  # UNK
    ('position_x', GameVariable.POSITION_X),
    ('position_y', GameVariable.POSITION_Y),
    ('position_z', GameVariable.POSITION_Z),
    # ('velocity_x', GameVariable.VELOCITY_X),
    # ('velocity_y', GameVariable.VELOCITY_Y),
    # ('velocity_z', GameVariable.VELOCITY_Z)
    ]

game_variables = dict(game_variables)


class DoomEnvironment():

  def __init__(self, cfgs):

    self.game = DoomGame()

    config_file = Path(cfgs['game_config'])
    scenario_file = Path(cfgs['scenario_config'])
    rewarded_game_vars = cfgs['rewarded_game_vars']
    reward_override = cfgs['reward_override']

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
    # self.game.add_available_button(Button.MOVE_LEFT)
    # self.game.add_available_button(Button.MOVE_RIGHT)
    # self.game.add_available_button(Button.ATTACK)

    # Adds game variables that will be included in state.
    # self.game.add_available_game_variable(GameVariable.HEALTH)

    # Causes episodes to finish after 2100 tics (actions)
    self.game.set_episode_timeout(2100)

    # Makes episodes start after 10 tics (~after raising the weapon)
    self.game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    self.game.set_window_visible(True)

    # Turns on the sound. (turned off by default)
    self.game.set_sound_enabled(True)

    # Sets the livin reward (for each move)
    # self.game.set_living_reward(1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR,
    # PLAYER mode is default)
    self.game.set_mode(Mode.PLAYER)

    self.game.init()

    self.action_size = self.game.get_available_buttons_size()
    self.actions = [a.tolist() for a in np.eye(self.action_size, dtype=bool)]

    self.reward_values = default_reward_values
    self.reward_values.update(reward_override)
    # list of game vars to be accounted final reward
    self.rewarded_game_vars = rewarded_game_vars
    # list of game vars available
    available_vars = self.game.get_available_game_variables()

    for game_var in self.rewarded_game_vars:
      assert game_variables[game_var] in available_vars, \
          "{} not available in this game".format(game_var)

    self.game_var_values = {}
    self.rewards = []
    logger.debug('Action space size {}'.format(self.action_size))
    logger.info('Environment setup')

  def make_action(self, action):

    reward = self.game.make_action(self.actions[action])
    current_game_vars = self.get_game_vars()

    state = ['BASE_REWARD', 'DEATH'][self.game.is_player_dead()]
    reward = self.reward_values[state]

    for var in self.game_var_values.keys():
      if var == 'health':
        delta_health = current_game_vars[var] - self.game_var_values[var]
        reward += np.clip(delta_health, self.reward_values['INJURED'],
                          self.reward_values['MEDIKIT'])

    self.game_var_values = current_game_vars

    self.rewards.append(reward)

    return reward

  def reset(self):

    self.rewards = []
    self.game.new_episode()
    self.game_var_values = self.get_game_vars()

  def get_game_vars(self):

    return {var: self.game.get_game_variable(game_variables[var])
            for var in self.rewarded_game_vars}

  def get_screen_buffer(self):

    return self.game.get_state().screen_buffer

  def is_episode_finished(self):

    return self.game.is_episode_finished()

  def is_player_dead(self):

    return self.game.is_player_dead()

  def get_total_reward(self):

    return np.sum(self.rewards)
