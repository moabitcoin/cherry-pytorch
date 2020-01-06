from cherry.envs.atari import AtariEnvironment
from cherry.envs.doom import DoomEnvironment
from cherry.envs.classic_control import ClassicControlEnvironment

envs_dict = {'atari': AtariEnvironment,
             'doom': DoomEnvironment,
             'classic_control': ClassicControlEnvironment}
