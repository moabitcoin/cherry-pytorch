from collections import OrderedDict

from cherry.agents.models import ConvNetS, ConvNetM, ConvNetL, MLP
from cherry.agents.algorithms import DQN, DDQN, VPG
from utils.helpers import get_logger

logger = get_logger(__name__)

MODELS = OrderedDict({None: None,
                      'convnet-small': ConvNetS,
                      'convnet-medium': ConvNetM,
                      'convnet-large': ConvNetL,
                      'mlp': MLP})

ALGOS = OrderedDict({None: None,
                     'dqn': DQN,
                     'ddqn': DDQN,
                     'vpg': VPG})


def get_model(model_type):

  model = None

  try:
    model = MODELS.get(model_type)

  except Exception as err:
    logger.error('Error setting up model {}, {}'.format(model_type, err))

  return model


def get_algo(algo_type):

  algo = None

  try:
    algo = ALGOS.get(algo_type)

  except Exception as err:
    logger.error('Error setting up algo {}, {}'.format(algo_type, err))

  return algo
