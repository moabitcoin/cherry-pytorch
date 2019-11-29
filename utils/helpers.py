import sys
import logging
from pathlib import Path

import yaml

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

  config_file = Path(config_file)

  if not config_file.is_file():
    logger.error('Not a file, {}'.format(config_file))
    return

  try:

    with config_file.open('r') as pfile:
      d = yaml.load(pfile)

    return d
  except Exception as err:
    logger.error('Error reading {}, {}'.format(config_file, err))
