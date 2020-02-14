import gym
import random
import argparse

import random
import tqdm
import numpy as np

from prettytable import PrettyTable

from utils.helpers import get_logger

logger = get_logger(__file__)


def reset_env(env):

  return env.reset()


def render_env(env):

  return env.render()


class FrozenLakeAgent():

  def __init__(self, n_state, n_action,
               max_eps=1.0, min_eps=0.01, eps_decay=0.005,
               lr=0.8, gamma=0.95):

    # current state of the agent
    self.state = None
    # Q-table numpy array
    self.q_table = None
    # current cummulative reward
    self.rewards = None

    # state space size
    self.n_state = n_state
    # action space size
    self.n_action = n_action
    # maximum exploration probability
    self.max_eps = max_eps
    # minimum exploration probability
    self.min_eps = min_eps
    # exploration probability decay
    self.eps_decay = eps_decay
    # agent learning rate
    self.lr = lr
    # reward discount
    self.gamma = gamma
    # current exploration probablity
    self.eps = self.max_eps

    self.reset_agent()
    self.reset_qtable()

  def __str__(self):

    return 'lr={}, state_size={}x{}, gamma={}'.format(self.lr,
                                                      self.n_state,
                                                      self.n_action,
                                                      self.gamma)

  def reset_agent(self):

    self.rewards = []

  def reset_qtable(self):

    self.q_table = np.zeros([self.n_state, self.n_action])

  def set_state(self, state):

    self.state = state

  def get_action(self, deterministic=False):

    if deterministic:
      return np.argmax(self.q_table[self.state, :])

    if random.random() < self.eps:
      return random.sample(range(self.n_action), 1)[0]
    else:
      return np.argmax(self.q_table[self.state, :])

  def update_qtable(self, action, next_state):

    reward = self.rewards[-1]

    target = reward + self.gamma * np.max(self.q_table[next_state, :])
    value = self.q_table[self.state, action]

    update = target - value

    self.q_table[self.state, action] += self.lr * (update)

  def update_reward(self, reward):

    self.rewards.append(reward)

  def decay_exploration(self, episode):

    self.eps = self.min_eps + (self.max_eps - self.min_eps) \
        * np.exp(-self.eps_decay * episode)

  def show_qtable(self):

    t = PrettyTable()
    t.field_names = ['state'] + ['action {}'.format(a)
                                 for a in range(self.n_action)]

    for idx, row in enumerate(self.q_table):
      t.add_row(['state {}'.format(idx)] + row.tolist())

    logger.info(t)


def solve_frozen_lake(n_train_episodes, max_steps, n_test_episodes=1,
                      show=False, play=False, render=False):

  env = gym.make("FrozenLake-v0")

  n_state = env.observation_space.n
  n_action = env.action_space.n

  agent = FrozenLakeAgent(n_state, n_action, max_eps=1.0, min_eps=0.01,
                          eps_decay=0.005, lr=0.8, gamma=0.95)
  logger.info('FrozenLakeAgent setup')
  logger.info(agent)

  ep_bar = tqdm.tqdm(range(n_train_episodes))

  for ep in ep_bar:

    state = reset_env(env)
    agent.set_state(state)

    for step in range(max_steps):

      if render:
        render_env(env)

      action = agent.get_action()

      next_state, reward, done, info = env.step(action)

      agent.update_reward(reward)
      agent.update_qtable(action, next_state)

      agent.set_state(next_state)

      if done:
        break

    agent.decay_exploration(ep)
    agent.reset_agent()

  if show:
    agent.show_qtable()

  if play:
    episode_rewards = []
    logger.info('Playing FrozenLake-V0 with the trained Agent')
    test_ep_bar = tqdm.tqdm(range(n_test_episodes))

    for ep in test_ep_bar:

      agent.reset_agent()
      state = env.reset()
      agent.set_state(state)

      for step in range(max_steps):

        action = agent.get_action(deterministic=True)

        next_state, reward, done, info = env.step(action)

        agent.update_reward(reward)
        agent.set_state(next_state)

        if done:
          episode_rewards.append(np.sum(agent.rewards))
          break

    av_score = np.sum(episode_rewards) / n_test_episodes
    logger.info('Av Score {0} runs {1:.4f}'.format(n_test_episodes, av_score))


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Frozen Lake OpenAI GYM')
  parser.add_argument('-n', dest='n_train_episodes', type=int,
                      help='Number of episodes', default=10000)
  parser.add_argument('-s', dest='max_steps', type=int,
                      help='Max steps per episode', default=99)
  parser.add_argument('-q', dest='show', action='store_true', default=False,
                      help='Show Q-Table at the end of training')
  parser.add_argument('-p', dest='play', action='store_true', default=False,
                      help='Play with the trained agent')
  parser.add_argument('-t', dest='n_test_episodes', type=int,
                      help='Number of test episodes', default=100)
  parser.add_argument('-r', dest='render', action='store_true',
                      help='Render visualisation', default=False)
  args = parser.parse_args()

  solve_frozen_lake(args.n_train_episodes, args.max_steps,
                    n_test_episodes=args.n_test_episodes,
                    show=args.show, play=args.play)
