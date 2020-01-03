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
from utils.helpers import read_yaml, get_logger, get_repo_hexsha, copy_yaml, \
    write_model


logger = get_logger(__file__)


def train_agent_of_atari(config_file, device='gpu'):

  hexsha = get_repo_hexsha()

  cuda_available = torch.cuda.is_available()
  cuda_and_device = cuda_available and device == 'gpu'

  if cuda_and_device:
    logger.info('Running CUDA benchmarks, GPU(s) device available')
  else:
    logger.info('Running on CPU(s)')

  device = torch.device('cuda' if cuda_and_device else 'cpu')

  cfgs = read_yaml(config_file)

  train_cfgs = cfgs['train']
  batch_size = train_cfgs['batch_size']
  update_target = train_cfgs['update_target']
  save_model = train_cfgs['save_model']
  model_dest = train_cfgs['model_dest']
  train_eps = train_cfgs['n_train_episodes']
  max_steps = train_cfgs['max_steps']
  policy_update = train_cfgs['policy_update']

  model_dest = Path(model_dest)

  env = AtariEnvironment(cfgs['env'])
  agent = AgentOfAtari(cfgs['agent'], action_size=env.action_size,
                       device=device)

  model_dest.mkdir(parents=True, exist_ok=True)
  copy_yaml(config_file, model_dest, hexsha)

  assert env.action_size == agent.action_size, \
      "Environment and state action size should match"

  train_ep = tqdm.tqdm(range(train_eps), ascii=True, unit='ep', leave=False)

  global_step = 0

  for ep in train_ep:

    agent.reset()
    frame = env.reset()

    agent.append_state(frame)

    train_step = tqdm.tqdm(range(max_steps), ascii=True,
                           unit='stp', leave=False)

    for step in train_step:

      global_step = ep * max_steps + step
      agent.set_eps(global_step)

      state = agent.get_state()
      action = agent.get_action(state)
      next_state, reward, done, info = env.step(action)
      agent.update_scores(reward)

      if done:
        next_state = env.reset()

      if info['ale.lives'] == 0:
        agent.show_score(train_step, global_step)
        agent.flush_episode()

      agent.append_state(next_state)
      states = agent.get_state(complete=True)
      agent.push_to_memory(states, action, reward, done)

      if global_step % policy_update == 0:
        agent.optimize(batch_size=batch_size)

      if global_step % update_target == 0:
        agent.update_target(global_step)

      if global_step % save_model == 0:
        tag = '{0:09d}-{1}'.format(global_step, hexsha)
        write_model(agent.policy, tag, model_dest)

    train_ep.set_description('Ep : {0}, Best Reward : {1:.3f}, '
                             'Eps : {2:.4f}'.format(ep, agent.top_scr,
                                                    agent.eps))

  tag = 'final-{1}'.format(hexsha)
  write_model(agent.policy, tag, model_dest)


if __name__ == '__main__':

  parser = argparse.ArgumentParser('Train an RL Agent to'
                                   ' play Atari Game (DQN)')
  parser.add_argument('-x', dest='config_file', type=str,
                      help='Config for the Atari env/agent', required=True)
  parser.add_argument('-d', dest='device', choices=['gpu', 'cpu'],
                      help='Device to run the train/test', default='gpu')

  args = parser.parse_args()

  train_agent_of_atari(args.config_file, device=args.device)
