from pathlib import Path

import tqdm
import torch
import numpy as np

from cherry.envs import setup_env
from cherry.agents import get_model, get_algo
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

    save_model = train_cfgs['save_model']
    model_dest = train_cfgs['model_dest']
    train_eps = train_cfgs['n_train_episodes']
    max_steps = train_cfgs['max_steps']
    env_solved = train_cfgs['env_solution']

    model_dest = Path(model_dest)

    env = setup_env(env_cfgs)
    model = get_model(agent_cfgs['model_type'])
    algo = get_algo(agent_cfgs['algo_type'])

    assert None not in [model, algo], "Model/Algo not setup"

    agent = algo(agent_cfgs, model=model,
                 action_size=env.action_size, device=device)

    model_dest.mkdir(parents=True, exist_ok=True)
    copy_yaml(config_file, model_dest, gitsha)

    assert env.action_size == agent.action_size, \
        "Environment's and Agent's state action size should match"

    train_ep = tqdm.tqdm(range(train_eps), ascii=True, unit='ep', leave=True)

    running_reward = 10

    for ep in train_ep:

      agent.reset()
      state = env.reset()

      agent.append_state(state)

      train_step = tqdm.tqdm(range(max_steps), ascii=True,
                             unit='stp', leave=False)

      for step in train_step:

        state = agent.get_state()
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        agent.append_reward(reward)
        agent.append_state(next_state)

        if done:

          running_reward = 0.05 * agent.get_episode_rewards() + \
              (1 - 0.05) * running_reward

          agent.append_episode_reward(running_reward)

          agent.discount_episode()
          agent.flash_episode()
          state = env.reset()
          agent.append_state(state)

      loss = agent.optimize()

      mean_rewards = np.mean(agent.ep_rewards)
      train_ep.set_description('Average reward: {:.3f}'.format(mean_rewards))

      if ep % save_model == 0:
        tag = '{0:09d}-{1}'.format(ep * max_steps, gitsha)
        write_model(agent.policy, tag, model_dest)

      best_reward = np.max(agent.ep_rewards)
      if best_reward >= env_solved:
        logger.info('Solved! At epside {}'
                    ' reward {:.3f} > {:.3f}'.format(ep, best_reward,
                                                     env_solved))
        break

    tag = 'final-{0}'.format(gitsha)
    write_model(agent.policy, tag, model_dest)
