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


def setup_model(cfgs, state_size, action_size, device):

  try:

    model = MODELS.get(cfgs['model_type'])
    model(cfgs, state_size, action_size, device)
    return model

  except Exception as err:
    logger.error('Error setting up model {}'.format(cfgs['model_type']))


def setup_algo(cfgs, action_size, device):

  try:

    model = ALGOS.get(cfgs['algo_type'])
    model(cfgs, state_size, action_size, device)
    return model

  except Exception as err:
    logger.error('Error setting up model {}'.format(cfgs['model_type']))
