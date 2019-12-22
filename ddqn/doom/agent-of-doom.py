#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse

import tqdm
import torch
import numpy as np

from ddqn.doom.environment import DoomEnvironment
from ddqn.doom.agent import AgentOfDoom
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
  max_steps = train_cfgs['max_steps']
  policy_update = train_cfgs['policy_update']
  n_buffer_episodes = train_cfgs['n_buffer_episodes']

  os.makedirs(model_dest, exist_ok=True)
  shutil.copy(config_file, model_dest)

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

  train_ep = tqdm.tqdm(range(train_eps), ascii=True,
                       unit='episode', leave=False)

  global_step = 0

  for ep in train_ep:

    env.reset()
    agent.reset()

    next_states = [env.get_frame() for _ in range(agent.state_size)]
    agent.append_states(next_states)

    train_step = tqdm.tqdm(range(max_steps), ascii=True,
                           unit='stp', leave=False)

    train_step.set_description('{0}, Reward : {1:.3f}, Eps : {2:.4f}, '
                               'Buffer {3}'.format(0, 0, agent.eps,
                                                   len(agent.replay)))

    for step in train_step:

      global_step = ep * max_steps + step
      agent.set_eps(global_step)

      state = agent.get_state()
      action = agent.get_action(state)
      next_state, reward, done = env.step(action)

      agent.append_states(next_states)
      states = agent.get_state(complete=True)
      agent.push_to_memory(states, action, reward, done)

      if done:

        total_score = env.get_total_reward()

        env.reset()
        agent.reset()

        next_states = [env.get_frame() for _ in range(agent.state_size)]
        agent.append_states(next_states)

        train_step.set_description('{0}, Reward : {1:.3f}, Eps : {2:.4f}, '
                                   'Buffer : {3}'.format(step, total_score,
                                                         agent.eps,
                                                         len(agent.replay)))

      if global_step % policy_update == 0 and ep > n_buffer_episodes:
        agent.optimize(batch_size=batch_size)

      if global_step % update_target == 0:
        agent.update_target(global_step)

      if global_step % save_model == 0:
        agent.save_model('{0:09d}'.format(global_step), model_dest)

  agent.save_model('final', model_dest)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train Agent of Doom with RL (DDQN')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config file for the Doom env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_doom(args.config_file, device=args.device)
