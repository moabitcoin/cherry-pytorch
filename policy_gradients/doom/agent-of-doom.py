#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

import tqdm
import torch
import numpy as np

from policy_gradients.doom.environment import DoomEnvironment
from policy_gradients.doom.agent import AgentOfDoom
from utils.helpers import read_yaml, get_logger


logger = get_logger(__file__)


def train_agent_of_doom(config_file, device='gpu'):

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
                      device=device)

  train_cfgs = cfgs['train']

  save_model = train_cfgs['save_model']
  model_dest = train_cfgs['model_dest']
  train_eps = train_cfgs['n_train_episodes']
  max_steps = train_cfgs['max_steps']

  os.makedirs(model_dest, exist_ok=True)
  shutil.copy(config_file, model_dest)

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

  train_ep = tqdm.tqdm(range(train_eps), ascii=True,
                       unit='episode', leave=False)

  global_step = 0

  for ep in train_ep:

    agent.reset()
    frame = env.reset()

    agent.append_state(frame)

    train_step = tqdm.tqdm(range(max_steps), ascii=True,
                           unit='stp', leave=False)

    done = False

    for step in train_step:

      global_step = ep * max_steps + step

      state = agent.get_state()
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)
      agent.append_state(next_state)
      agent.append_reward(reward)

      if global_step % save_model == 0:
        agent.save_model('{0:09d}'.format(global_step), model_dest)

      if done:
        agent.append_episode_score(env.get_total_reward())
        agent.discount_episode()
        agent.flash_episode()
        frame = env.reset()
        agent.append_state(frame)

    # if the last episod didnt finish
    if not done:
      agent.discount_episode()

    rewards = agent.get_total_rewards()
    loss = agent.optimize()
    train_ep.set_description('{} Loss {:.6f}, Reward {:.3f}'.format(ep,
                                                                    loss,
                                                                    rewards))

  agent.save_model('final', model_dest)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train Agent of Doom with RL (DQN)')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config file for the Doom env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_doom(args.config_file, device=args.device)
