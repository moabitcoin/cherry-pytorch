from pathlib import Path

from cherry.agents.models import model_dict
from utils.helpers import add_verbosity_parser, read_yaml, copy_yaml, \
    get_repo_hexsha


class Trainer:

  def __init__(self):

    self.cfgs = None
    self.trainer = None
    self.gitsha = None

  def build_parser(self, parser):

    parser.add_argument('-c', '--config_file', type=Path,
                        help='Path to Config file', required=True)

    parser = add_verbosity_parser(parser)

  def run(self, config_file):

    self.gitsha = get_repo_hexsha()
    self.cfgs = read_yaml(config_file)

    assert self.cfgs is not None, 'Error reading config file'
