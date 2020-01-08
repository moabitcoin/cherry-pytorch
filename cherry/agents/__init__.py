from collections import OrderedDict

from cherry.agents.models import ConvNetS, ConvNetM, ConvNetL, MLP, ReplayBuffer
from cherry.agents.algorithms import DQN, DDQN, VPG
from utils.helpers import get_logger

logger = get_logger(__file__)

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
    logger.info('Setting up {}'.format(model_type))

  except Exception as err:
    logger.error('Error setting up model {}, {}'.format(model_type, err))

  return model


def build_agent(cfgs, **kwargs):

  agent = None

  try:
    agent = ALGOS.get(cfgs['agent_type'])(cfgs, **kwargs)

  except Exception as err:
    logger.error('Setting up algo {}, {}'.format(cfgs['agent_type'], err))

  return agent
