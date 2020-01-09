import sys
import logging
import argparse
import shutil
from collections import OrderedDict

import git
import yaml
import torch
import torch.optim as optim
from baselines.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv

OPTS = OrderedDict({None: None,
                    'adam': optim.Adam,
                    'rmsprop': optim.RMSprop})

CLI_LOGGING_FORMAT = '[%(filename)s][%(funcName)s:%(lineno)d]' + \
    '[%(levelname)s] %(message)s'
CLI_LOGGING_STREAM = sys.stdout


def get_logger(logger_name, log_level='info'):

  CLI_LOGGING_LEVEL = getattr(logging, log_level.upper(), None)

  logger = logging.getLogger(logger_name)
  logger.setLevel(CLI_LOGGING_LEVEL)
  ch = logging.StreamHandler(CLI_LOGGING_STREAM)
  formatter = logging.Formatter(CLI_LOGGING_FORMAT)
  ch.setFormatter(formatter)
  ch.setLevel(CLI_LOGGING_LEVEL)
  logger.addHandler(ch)
  logger.propagate = False

  return logger


logger = get_logger(__file__)


def read_yaml(config_file):

  if not config_file.is_file():
    logger.error('Not a file, {}'.format(config_file))
    return

  try:

    with config_file.open('r') as pfile:
      d = yaml.load(pfile, yaml.FullLoader)

    assert validate_config(d)

    logger.info('Read config file {}'.format(config_file.as_posix()))

    return d
  except Exception as err:
    logger.error('Error reading {}, {}'.format(config_file, err))


def atari_play_env(env):

  env = EpisodicLifeEnv(env)

  if 'FIRE' in env.unwrapped.get_action_meanings():
    env = FireResetEnv(env)

  return env


def get_repo_hexsha():

  filepath = __file__
  repopath = filepath.split('utils')[0]

  g = git.Repo(repopath)

  return g.head.commit.hexsha[:8]


def copy_yaml(src_file, dest_dir, hexsha):

  stem = src_file.stem
  fname = '{}-{}.yaml'.format(stem, hexsha)
  dst_file = dest_dir.joinpath(fname)

  shutil.copyfile(src_file.as_posix(), dst_file.as_posix())


def write_model(model, tag, dest):

  model_savefile = '{0}/agent-{1}.pth'.format(dest, tag)
  logger.debug("Saving Agent to {}".format(model_savefile))

  torch.save(model.state_dict(), model_savefile)


def add_verbosity_parser(parser):

  parser.add_argument('-l', '--log', dest='log', choices=['info', 'debug'],
                      default='info', help='Set verbosity for the logger')

  return parser


def validate_config(cfgs):

  assert cfgs.get('env'), 'Expected Environment info in config file'
  assert cfgs.get('agent'), 'Expected Agent info in config file'
  assert cfgs.get('train'), 'Expected Training info in config file'
  assert cfgs.get('test'), 'Expected Testing info in config file'

  return True
