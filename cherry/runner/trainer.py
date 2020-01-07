from pathlib import Path

import tqdm
import torch
import numpy as np

from cherry.envs import build_env
from cherry.agents import get_model, build_agent
from utils.helpers import add_verbosity_parser, read_yaml, copy_yaml, \
    get_repo_hexsha, validate_config, get_logger, write_model


logger = get_logger(__file__)


class Trainer:

  def __init__(self):

    pass

  def build_parser(self, parser):

    parser.add_argument('-c', '--config_file', type=Path,
                        help='Path to Config file', required=True)

    parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                        help='Device to run the train/test', default='gpu')
    parser.set_defaults(main=self._run)

    parser = add_verbosity_parser(parser)

  def _run(self, args):

    config_file = args.config_file
    device = args.device

    gitsha = get_repo_hexsha()

    cuda_available = torch.cuda.is_available()
    cuda_and_device = cuda_available and device == 'gpu'

    device = torch.device('cuda' if cuda_and_device else 'cpu')

    if cuda_and_device:
      logger.info('Running CUDA benchmarks, GPU(s) device available')
    else:
      logger.info('Running on CPU(s)')

    try:
      cfgs = read_yaml(config_file)
    except Exception as err:
      logger.error('Error reading config file {}, {}'.format(config_file, err))
      return

    env_cfgs = cfgs['env']
    agent_cfgs = cfgs['agent']
    train_cfgs = cfgs['train']

    model_dest = train_cfgs['model_dest']
    model_dest = Path(model_dest)

    env = build_env(env_cfgs)

    model = get_model(agent_cfgs['model_type'])
    agent = build_agent(agent_cfgs, model=model, device=device)

    model_dest.mkdir(parents=True, exist_ok=True)
    copy_yaml(config_file, model_dest, gitsha)

    assert env.action_size == agent.action_size, "Env ≠ Agent {} ≠ {} action' \
        ' size should match".format(env.action_size, agent.action_size)

    agent.train(env, train_cfgs, gitsha, model_dest)
