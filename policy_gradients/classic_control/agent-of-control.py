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

  env = ClassicControlEnvironment(cfgs['env'])
  agent = AgentOfControl(cfgs['agent'], action_size=env.action_size,
                         device=device)

  os.makedirs(model_dest, exist_ok=True)
  shutil.copy(config_file, model_dest)

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

  train_ep = tqdm.tqdm(range(train_eps), ascii=True, unit='ep', leave=False)

  global_step = 0
  running_reward = 10

  for ep in train_ep:

    agent.reset()
    state = env.reset()

    agent.append_state(state)

    step = 0
    train_step = tqdm.tqdm(range(max_steps), ascii=True,
                           unit='stp', leave=False)

    for step in train_step:

      global_step = ep * max_steps + step

      state = agent.get_state()
      action = agent.get_action(state)
      next_state, reward, done, info = env.step(action)
      agent.append_reward(reward)
      agent.append_state(next_state)

      if done:
        break

    running_reward = running_reward * 0.99 + step * 0.01
    loss = agent.optimize()

    if ep % save_model == 0:
      agent.save_model('{0:09d}'.format(global_step), model_dest)
      print('Episode {}\tLast length: {:5d}'
            '\tAverage length: {:.2f}\tLoss: '
            '{:.4f}'.format(ep, step, running_reward, loss))

  agent.save_model('final', model_dest)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train an RL Agent to'
                                   ' play Classic Control Problems (PG)')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config for the Atari env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_control(args.config_file, device=args.device)
