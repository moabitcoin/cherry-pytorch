#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse
from pathlib import Path

import tqdm
import torch
import numpy as np
import skvideo
import skvideo.io
from gym import wrappers

from policy_gradients.classic_control.environment import ClassicControlEnvironment
from policy_gradients.classic_control.agent import AgentOfControl
from utils.helpers import read_yaml, get_logger


logger = get_logger(__file__)


def play_control(config_file, model_file=None, device='gpu'):

  assert Path(model_file).is_file(), \
      'Model file {} does not exists'.format(model_file)

  cuda_available = torch.cuda.is_available()
  cuda_and_device = cuda_available and device == 'gpu'

  if cuda_and_device:
    logger.info('Running CUDA benchmarks, GPU(s) device available')
  else:
    logger.info('Running on CPU(s)')

  device = torch.device('cuda' if cuda_and_device else 'cpu')

  cfgs = read_yaml(config_file)

  test_cfgs = cfgs['test']
  state_dest = test_cfgs['state_dest']
  test_episodes = test_cfgs['n_test_episodes']

  if not Path(state_dest).is_dir():
    os.makedirs(state_dest)

  env = ClassicControlEnvironment(cfgs['env'])
  env.update_env(wrappers.Monitor, directory=state_dest, force=True)

  agent = AgentOfControl(cfgs['agent'], action_size=env.action_size,
                         device=device, model_file=model_file)

  test_ep = tqdm.tqdm(range(test_episodes), ascii=True, unit='episode')

  for ep in test_ep:

    agent.reset()
    frame = env.reset()

    agent.append_state(frame)

    test_steps = tqdm.tqdm(range(test_cfgs['max_steps']), ascii=True,
                           unit='episode', leave=False)

    for step in test_steps:

      state = agent.get_state()
      action = agent.get_action(state)
      next_state, reward, done, info = env.step(action)
      agent.append_state(next_state)
      agent.append_reward(reward)

      if done:
        ep_reward = agent.get_episode_rewards()
        test_ep.set_description('Ep : {0}, Reward : {1:.3f}'
                                .format(ep, ep_reward))
        break


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Test Agent of Atari(DQN)')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config file for the Atari agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')
  parser.add_argument('-m', dest='model_file', required=True,
                      help='Model to test with')

  args = parser.parse_args()

  play_control(args.config_file, model_file=args.model_file, device=args.device)
