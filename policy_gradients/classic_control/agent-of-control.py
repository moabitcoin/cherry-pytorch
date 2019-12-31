#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

import tqdm
import torch
import numpy as np

from policy_gradients.classic_control.environment import ClassicControlEnvironment
from policy_gradients.classic_control.agent import AgentOfControl
from utils.helpers import read_yaml, get_logger


logger = get_logger(__file__)


def train_agent_of_control(config_file, device='gpu'):

  cuda_available = torch.cuda.is_available()
  cuda_and_device = cuda_available and device == 'gpu'

  if cuda_and_device:
    logger.info('Running CUDA benchmarks, GPU(s) device available')
  else:
    logger.info('Running on CPU(s)')

  device = torch.device('cuda' if cuda_and_device else 'cpu')

  cfgs = read_yaml(config_file)

  train_cfgs = cfgs['train']
  save_model = train_cfgs['save_model']
  model_dest = train_cfgs['model_dest']
  train_eps = train_cfgs['n_train_episodes']
  max_steps = train_cfgs['max_steps']
  env_solved = train_cfgs['env_solution']

  env = ClassicControlEnvironment(cfgs['env'])
  agent = AgentOfControl(cfgs['agent'], action_size=env.action_size,
                         device=device)

  os.makedirs(model_dest, exist_ok=True)
  shutil.copy(config_file, model_dest)

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

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
      agent.save_model('{0:09d}'.format(ep * max_steps), model_dest)

    best_reward = np.max(agent.ep_rewards)
    if best_reward >= env_solved:
      logger.info('Solved! At epside {}'
                  ' reward {:.3f} > {:.3f}'.format(ep, best_reward,
                                                   env_solved))
      break

  agent.save_model('final', model_dest)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train an RL Agent to solve Classic '
                                   'Control problems (with PG)')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config for the Atari env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_control(args.config_file, device=args.device)
