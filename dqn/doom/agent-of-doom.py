import argparse

import tqdm
import torch

from dqn.doom.environment import DoomEnvironment
from dqn.doom.agent import AgentOfDoom
from utils.helpers import read_yaml, get_logger


logger = get_logger(__file__)


def train_agent_of_doom(config_file, show=False, play=False, device='gpu'):

  cuda_available = torch.cuda.is_available()
  if cuda_available and device is 'gpu':
    logger.info('Running CUDA benchmarks, GPU(s) device available')
  else:
    logger.info('Running on CPU(s)')

  device = torch.device('cuda' if cuda_available and device is 'gpu' else 'cpu')

  cfgs = read_yaml(config_file)

  env = DoomEnvironment(cfgs['env'])
  agent = AgentOfDoom(cfgs['agent'], device=device)
  train_cfgs = cfgs['train_cfgs']
  batch_size = train_cfgs['batch_size']

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

  train_ep = tqdm.tqdm(train_cfgs['n_train_episodes'])

  for ep in train_ep:

    env.game.new_episode()
    agent.restart()

    frame = env.game.get_state().screen_buffer
    agent.set_history(frame, new_episode=True)

    for step in range(cfgs['max_steps']):

      agent.set_eps(step)

      state = agent.get_history()
      action = agent.get_action(state)
      reward = env.game.make_action(action)

      done = game.is_episode_finished()

      next_frame = None if done \
          else env.game.get_state().screen_buffer

      agent.set_history(next_frame)
      next_state = agent.get_history()
      agent.push_to_memory([state, action, next_state, reward])

      agent.update(batch_size=batch_size)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train Agent of Doom with RL')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config file for the Doom env/agent', required=True)
  parser.add_argument('-n', dest='n_train_episodes', type=int,
                      help='Number of episodes', default=50000)
  parser.add_argument('-s', dest='max_steps', type=int,
                      help='Max steps per episode', default=99)
  parser.add_argument('-q', dest='show', action='store_true', default=False,
                      help='Show Q-Table at the end of training')
  parser.add_argument('-p', dest='play', action='store_true', default=False,
                      help='Play with the trained agent')
  parser.add_argument('-t', dest='n_test_episodes', type=int,
                      help='Number of test episodes', default=100)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_doom(args.config_file, show=args.show, play=args.play,
                      device=args.device)
