from pathlib import Path

from cherry.envs import setup_env
from cherry.agents import setup_model, setup_algo
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

    parser = add_verbosity_parser(parser)

  def run(self, config_file):

    gitsha = get_repo_hexsha()

    cuda_available = torch.cuda.is_available()
    cuda_and_device = cuda_available and device == 'gpu'

    if cuda_and_device:
      logger.info('Running CUDA benchmarks, GPU(s) device available')
    else:
      logger.info('Running on CPU(s)')

    device = torch.device('cuda' if cuda_and_device else 'cpu')

    try:
      cfgs = read_yaml(config_file)
    except Exception as err:
      logger.error('Error reading config file {}, {}'.format(config_file, err))

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
    agent = setup_agent(agent_cfgs)

    model_dest.mkdir(parents=True, exist_ok=True)
    copy_yaml(config_file, model_dest, gitsha)

    assert self.env.action_size == self.agent.action_size, \
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
        tag = '{0:09d}-{1}'.format(ep * max_steps, hexsha)
        write_model(agent.policy, tag, model_dest)

      best_reward = np.max(agent.ep_rewards)
      if best_reward >= env_solved:
        logger.info('Solved! At epside {}'
                    ' reward {:.3f} > {:.3f}'.format(ep, best_reward,
                                                     env_solved))
        break

    tag = 'final-{0}'.format(hexsha)
    write_model(agent.policy, tag, model_dest)
