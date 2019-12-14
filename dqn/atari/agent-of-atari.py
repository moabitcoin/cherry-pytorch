#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

import tqdm
import torch
import numpy as np

from dqn.atari.environment import AtariEnvironment
from dqn.atari.agent import AgentOfAtari
from utils.helpers import read_yaml, get_logger


logger = get_logger(__file__)


def train_agent_of_atari(config_file, device='gpu'):

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
                       device=device)

  train_cfgs = cfgs['train']

  batch_size = train_cfgs['batch_size']
  update_target = train_cfgs['update_target']
  save_model = train_cfgs['save_model']
  model_dest = train_cfgs['model_dest']
  train_eps = train_cfgs['n_train_episodes']

  os.makedirs(model_dest, exist_ok=True)
  shutil.copy(config_file, model_dest)

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

  train_ep = tqdm.tqdm(range(train_eps), ascii=True, unit='ep', leave=False)

  for ep in train_ep:

    agent.reset()
    frame = env.game.reset()

    agent.set_history(frame, new_episode=True)

    train_step = tqdm.tqdm(range(train_cfgs['max_steps']), ascii=True,
                           unit='stp', leave=False)

    for step in train_step:

      agent.set_eps((ep + 1) * step)

      state = agent.get_history()
      action = agent.get_action(state)
      next_state, reward, done, info = env.game.step(env.actions[action])
      agent.update_scores(reward)

      next_frame = None if done else next_state

      agent.set_history(next_frame)
      next_state = agent.get_history()
      agent.push_to_memory(state, action, next_state, reward)

      agent.update(batch_size=batch_size)

      if done:

        frame = env.game.reset()

        agent.set_history(frame, new_episode=True)
        agent.show_score(train_step)

        agent.restart()

    train_ep.set_description('Best Reward : {0:.3f}'
                             'Eps : {1:.4f}'.format(agent.top_scr,
                                                    agent.eps))
    if ep % update_target == 0:
      agent.update_target(ep)

    if ep % save_model == 0:
      agent.save_model('{0:06d}'.format(ep), model_dest)

  agent.save_model('final', model_dest)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train an RL Agent to play Atari Game')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config for the Atari env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_atari(args.config_file, device=args.device)
