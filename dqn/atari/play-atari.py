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

from dqn.atari.environment import AtariEnvironment
from dqn.atari.agent import AgentOfAtari
from utils.helpers import read_yaml, get_logger


logger = get_logger(__file__)


def play_atari(config_file, model_file=None, device='gpu'):

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

  env = AtariEnvironment(cfgs['env'])
  agent = AgentOfAtari(cfgs['agent'], action_size=env.action_size,
                       device=device, model_file=model_file)

  test_cfgs = cfgs['test']
  state_dest = test_cfgs['state_dest']
  test_episodes = test_cfgs['n_test_episodes']

  test_ep = tqdm.tqdm(range(test_episodes), ascii=True, unit='episode')

  if not Path(state_dest).is_dir():
    os.makedirs(state_dest)

  for ep in test_ep:

    vid_file = '{0}/states-ep-{1:06d}.mp4'.format(state_dest, ep)

    writer = vid_writer(vid_file, outputdict={'-vcodec': 'h264',
                                              '-b': '300000000'})

    agent.reset()
    env.game.new_episode()

    # no exploration
    agent.eps = 0.0

    frame = env.game.get_state().screen_buffer
    agent.set_history(frame, new_episode=True)

    writer.writeFrame(frame)

    for step in range(test_cfgs['max_steps']):

      state = agent.get_history()
      action = agent.get_action(state)
      reward = env.game.make_action(env.actions[action])

      done = env.game.is_episode_finished()

      next_frame = None if done \
          else env.game.get_state().screen_buffer

      if next_frame is not None:
        writer.writeFrame(next_frame)

      agent.set_history(next_frame)

      if done:
        agent.update_scores(env.game.get_total_reward())

        agent.restart()
        env.game.new_episode()

        frame = env.game.get_state().screen_buffer
        agent.set_history(frame, new_episode=True)

    writer.close()

    mean_score = 0.0 if agent.scores == [] else np.mean(agent.scores)

    test_ep.set_description('Episode {0} Reward : {1:.3f}'.format(ep,
                                                                  mean_score))


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Test Agent of Atari(DQN)')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config file for the Atari env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')
  parser.add_argument('-m', dest='model_file', required=True,
                      help='Model to test with')

  args = parser.parse_args()

  play_atari(args.config_file, model_file=args.model_file, device=args.device)
