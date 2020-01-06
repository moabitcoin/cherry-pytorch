import sys
import logging
import argparse
import shutil

import git
import yaml
import torch
from baselines.common.atari_wrappers import EpisodicLifeEnv, FireResetEnv

CLI_LOGGING_FORMAT = '[%(filename)s][%(funcName)s:%(lineno)d]' + \
    '[%(levelname)s] %(message)s'
CLI_LOGGING_LEVEL = logging.INFO
CLI_LOGGING_STREAM = sys.stdout


def get_logger(logger_name):

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

  parser.add_argument('-v', '--verbose', action='store_true',
                      default=True, help='Only warnings')
  parser.add_argument('-vv', '--very-verbose', action='store_true',
                      default=False, help='Warnings + Info')
  parser.add_argument('-vvv', '--very-very-verbose', action='store_true',
                      default=False, help='Warning + Info + Debug')

  return parser


def validate_config(cfgs):

  assert cfgs.get('env') is not None, \
      'Expected Environment info in config file'
  assert cfgs.get('agent') is not None \
      'Expected Agent info in config file'
  assert cfgs.get('train') is not None \
      'Expected Training info in config file'
  assert cfgs.get('test') is not None \
      'Expected Testing info in config file'

  return True
