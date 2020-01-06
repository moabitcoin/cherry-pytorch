from collections import OrderedDict

from cherry.agents.models import ConvNetS, ConvNetM, ConvNetL, MLP
from cherry.agents.algorithms import DQN, DDQN, VPG

model_dict = OrderedDict({'convnet-small': ConvNetS,
                          'convnet-medium': ConvNetM,
                          'convnet-large': ConvNetL,
                          'mlp': MLP})

algo_dict = OrderedDict({'dqn': DQN,
                         'ddqn': DDQN,
                         'vpg': VPG})
