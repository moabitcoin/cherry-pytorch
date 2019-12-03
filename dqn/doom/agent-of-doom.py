#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

import tqdm
import torch
import numpy as np

from dqn.doom.environment import DoomEnvironment
from dqn.doom.agent import AgentOfDoom
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

  batch_size = train_cfgs['batch_size']
  update_target = train_cfgs['update_target']
  save_model = train_cfgs['save_model']
  model_dest = train_cfgs['model_dest']
  train_eps = train_cfgs['n_train_episodes']

  os.makedirs(model_dest, exist_ok=True)
  shutil.copy(config_file, model_dest)

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

  train_ep = tqdm.tqdm(range(train_eps), ascii=True, unit='episode')

  for ep in train_ep:

    agent.reset()
    env.game.new_episode()

    frame = env.game.get_state().screen_buffer
    agent.set_history(frame, new_episode=True)

    for step in range(train_cfgs['max_steps']):

      agent.set_eps(ep * step)

      state = agent.get_history()
      action = agent.get_action(state)
      reward = env.game.make_action(env.actions[action])

      done = env.game.is_episode_finished()

      next_frame = None if done \
          else env.game.get_state().screen_buffer

      agent.set_history(next_frame)
      next_state = agent.get_history()
      agent.push_to_memory(state, action, next_state, reward)

      agent.update(batch_size=batch_size)

      if done:
        agent.update_scores(env.game.get_total_reward())

        agent.restart()
        env.game.new_episode()

        frame = env.game.get_state().screen_buffer
        agent.set_history(frame, new_episode=True)

    mean_score = 0.0 if agent.scores == [] else np.mean(agent.scores)
    mean_loss = 0.0 if agent.losses == [] else np.mean(agent.losses)

    train_ep.set_description('Reward : {1:.3f} Loss : {2:.4f} eps'
                             ' : {3:.4f}'.format(ep, mean_score, mean_loss,
                                                 agent.eps))
    if ep % update_target == 0:
      agent.update_target(ep)

    if ep % save_model == 0:
      agent.save_model('{0:06d}'.format(ep), model_dest)

  agent.save_model('final', model_dest)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train Agent of Doom with RL')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config file for the Doom env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_doom(args.config_file, device=args.device)
