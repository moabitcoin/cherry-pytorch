import argparse

from cherry.runner import Trainer, Player


def run():

  trainer = Trainer()
  player = Player()

  Formatter = argparse.ArgumentDefaultsHelpFormatter

  parser = argparse.ArgumentParser(description='Cherry options for training/'
                                   'playing the agent')
  subparsers = parser.add_subparsers(title='Commands', dest='command',
                                     description='Valid command for Cherry',
                                     help='Select train/play mode')
  subparsers.required = True

  train_parser = subparsers.add_parser('train', help='🚆 Train the RL agent',
                                       formatter_class=Formatter)
  play_parser = subparsers.add_parser('play', help='🎮 Play the RL agent',
                                      formatter_class=Formatter)

  trainer.build_parser(train_parser)
  player.build_parser(play_parser)

  args = parser.parse_args()
  args.main(args)
