import sys
import logging

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
