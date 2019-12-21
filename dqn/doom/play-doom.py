#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import tqdm
import torch
import numpy as np
import skvideo
import skvideo.io
from skvideo.io import FFmpegWriter as vid_writer

from dqn.doom.environment import DoomEnvironment
from dqn.doom.agent import AgentOfDoom
from utils.helpers import read_yaml, get_logger


logger = get_logger(__file__)


def play_doom(config_file, model_file=None, device='gpu'):

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

  env = DoomEnvironment(cfgs['env'])
  agent = AgentOfDoom(cfgs['agent'], action_size=env.action_size,
                      device=device, model_file=model_file)

  test_cfgs = cfgs['test']
  state_dest = test_cfgs['state_dest']
  test_episodes = test_cfgs['n_test_episodes']
  max_steps = test_cfgs['max_steps']

  test_ep = tqdm.tqdm(range(test_episodes), ascii=True, unit='episode')

  if not Path(state_dest).is_dir():
    os.makedirs(state_dest)

  for ep in test_ep:

    vid_file = '{0}/states-ep-{1:09d}.mp4'.format(state_dest, ep)

    writer = vid_writer(vid_file, outputdict={'-vcodec': 'h264',
                                              '-b': '300000000'})

    agent.reset()
    env.reset()

    # no exploration
    agent.eps = 0.1

    frame = env.get_screen_buffer()
    agent.append_state(frame)

    writer.writeFrame(frame)

    test_step = tqdm.tqdm(range(max_steps), ascii=True, unit='stp')

    for step in test_step:

      state = agent.get_state()
      action = agent.get_action(state)
      reward = env.make_action(action)
      done = env.is_episode_finished()

      if done:

        reward = env.get_total_reward()

        agent.reset()
        env.reset()

        test_step.set_description('{0}/{1} Reward : {2:.3f}'.format(ep, step,
                                                                    reward))

      next_frame = env.get_screen_buffer()
      agent.append_state(next_frame)

      writer.writeFrame(next_frame)

    writer.close()


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Test Agent of Doom')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config file for the Doom env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')
  parser.add_argument('-m', dest='model_file', required=True,
                      help='Model to test with')

  args = parser.parse_args()

  play_doom(args.config_file, model_file=args.model_file, device=args.device)
