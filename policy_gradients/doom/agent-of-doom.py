#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
from pathlib import Path

import tqdm
import torch
import numpy as np

from policy_gradients.doom.environment import DoomEnvironment
from policy_gradients.doom.agent import AgentOfDoom
from utils.helpers import read_yaml, get_logger, get_repo_hexsha, copy_yaml, \
    write_model


logger = get_logger(__file__)


def train_agent_of_doom(config_file, model_file, device='gpu'):

  hexsha = get_repo_hexsha()

  cuda_available = torch.cuda.is_available()
  cuda_and_device = cuda_available and device == 'gpu'

  if cuda_and_device:
    logger.info('Running CUDA benchmarks, GPU(s) device available')
  else:
    logger.info('Running on CPU(s)')

  device = torch.device('cuda' if cuda_and_device else 'cpu')

  cfgs = read_yaml(config_file)

  env = DoomEnvironment(cfgs['env'])
  agent = AgentOfDoom(cfgs['agent'], action_size=env.action_size,
                      device=device, model_file=model_file)

  train_cfgs = cfgs['train']

  save_model = train_cfgs['save_model']
  model_dest = train_cfgs['model_dest']
  train_eps = train_cfgs['n_train_episodes']
  max_steps = train_cfgs['max_steps']
  env_solved = train_cfgs['env_solution']

  model_dest = Path(model_dest)

  model_dest.mkdir(parents=True, exist_ok=True)
  copy_yaml(config_file, model_dest, hexsha)

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

  train_ep = tqdm.tqdm(range(train_eps), ascii=True, unit='ep', leave=True)

  for ep in train_ep:

    agent.reset()
    frame = env.reset()

    agent.set_state(frame)

    train_step = tqdm.tqdm(range(max_steps), ascii=True,
                           unit='stp', leave=False)

    done = False

    for step in train_step:

      state = agent.get_state()
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      agent.append_reward(reward)
      agent.append_state(next_state)

      if done:
        ep_reward = env.get_total_reward()
        agent.append_episode_reward(ep_reward)

        agent.discount_episode()
        agent.flash_episode()
        state = env.reset()
        agent.set_state(state)

    if not done:
      ep_reward = env.get_total_reward()
      agent.append_episode_reward(ep_reward)
      agent.discount_episode()

    loss = agent.optimize()

    mean_rewards = np.mean(agent.ep_rewards)
    train_ep.set_description('Average reward: {:.3f} '
                             'Loss : {:.3f}'.format(mean_rewards, loss))

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


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train Agent of Doom with RL (PG)')
  parser.add_argument('-x', dest='config_file', type=Path, required=True,
                      help='Config file for the Doom env/agent')
  parser.add_argument('-m', dest='model_file', help='Model to test with')
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_doom(args.config_file, args.model_file, device=args.device)
