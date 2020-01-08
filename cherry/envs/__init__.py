from cherry.envs.atari import AtariEnvironment
from cherry.envs.doom import DoomEnvironment
from cherry.envs.classic_control import ClassicControlEnvironment
from utils.helpers import get_logger

logger = get_logger(__name__)

ENVS = {'atari': AtariEnvironment,
        'doom': DoomEnvironment,
        'classic_control': ClassicControlEnvironment}


def build_env(cfgs):

  try:

    env = ENVS.get(cfgs['type'])
    return env(cfgs)

  except Exception as err:
    logger.error('Error setting up env {}, {}'.format(cfgs['type'], err))
