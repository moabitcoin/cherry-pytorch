<h1 align='center'>:cherries: Cherry in PyTorch</h1>

Reinforcement Learning &amp; other bedtime stories in PyTorch. This repository is [pytorch](https://pytorch.org) implementation of the [Intro RL course](https://github.com/simoninithomas/Deep_reinforcement_learning_Course) & [Advanced RL course](https://github.com/dennybritz/reinforcement-learning). Before training/playing the agent within conda env, please install the [external dependencies](https://github.com/moabitcoin/cherry-pytorch#dependencies). The name cherry is derived from Yann Lecun's [statement](https://www.quora.com/What-do-you-think-about-reinforcement-learning-Is-it-the-cherry-on-the-cake-as-Yann-LeCun-puts-it-1) regarding RL being the cherry on the ML [cake.](https://miro.medium.com/max/1200/1*bvMhd_xpVxfJYoKXYp5hug.png).

# Table of Contents
* [Installation](#Installation)
  - [Conda](#snake-conda) or [Docker](#whale-docker)
* [Sample usage](#tada-usage)
  - [Cartpole-v0](#eyes-example)
* [Dependencies (Conda only)](#two_men_holding_hands-dependencies)
  - [Vizdoom](#godmode-vizdoom), [ALE](#space_invader-arcade-learning-environment), [Baselines](#chart_with_upwards_trend-openai-rl-baselines)
* [Architectures](#architectures)
* [Agents](#agents)
* [Environments](#environments)

## :computer: Installation
```
Tested with ubuntu:18.04 only
```
### :snake: Conda
```
# After installed external dependencies
conda create -n cherry python=3.6
conda activate cherry
pip install -r requirements.txt
pip install -e .
```
### :whale: Docker
```
# CPU
make install dockerfile=Dockerfile.cpu dockerimage=moabitcoin/cherry-pytorch-cpu
make run dockerimage=moabitcoin/cherry-pytorch-cpu
# GPU
make install dockerfile=Dockerfile.gpu
make gpu dockerimage=moabitcoin/cherry-pytorch-gpu
```

## :tada: Usage
```
# Inside the virtual environment or Docker
cherry --help
cherry train --help
```
```
usage: cherry train [-h] -c CONFIG_FILE [-d {gpu,cpu}] [-l {info,debug}]

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        Path to Config file (default: None)
  -d {gpu,cpu}          Device to run the train/test (default: gpu)
  -l {info,debug}, --log {info,debug}
                        Set verbosity for the logger (default: info)
```
### :eyes: Example
```
# Training the agent to balance the Cartpole
cherry train -c configs/control.yaml -d cpu
# Playing with the trained agent saved at <model_dest> in configs/control.yaml
cherry play -c <model_dest>/control-<commit-gitsha>.yaml -d cpu -m <model_dest>/agent-final-<commit-gitsha>.pth
# Visualising the agent performance from the videos saved under state_dest in configs/control.yaml
vlc <state_dest>
```

## Architectures
We support 4 [feedforward](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/agents/README.md#architectures) architectures within `cherry`. We plan to expand the list of architectures to include Recurrent/Transformer/Memory architectures.

## Agents
We support [3 Agents](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/agents/README.md#agents) within `cherry`. We plan to expand the list of agents to include TRPO/DDPG/PPO/A3C.

## Environments
We support [3 environments](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/envs/README.md) within cherry. This list would be expanded to include [Robotics](https://gym.openai.com/envs/#robotics) and other [3rd party](https://github.com/openai/gym/blob/master/docs/environments.md#third-party-environments) environments.

## :two_men_holding_hands: Dependencies
### :godmode: ViZDoom
Install dependencies as outlined [here](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps)
```
sudo apt-get install libcanberra-gtk-module
pip install vizdoom
# If on a headless machine
export DISPLAY=:0
```

### :space_invader: Arcade Learning Environment
```
https://github.com/openai/gym/blob/master/docs/environments.md#atari
```

### :chart_with_upwards_trend: OpenAI RL baselines
Needed for [Deepmind styled](https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L275) pre-processing wrappers. Needs TF>=1.14 :woman_facepalming:
```
pip install tensorflow_gpu==1.14
https://github.com/openai/baselines/tree/master#installation
```

## :tv: Remote Render
```
https://github.com/openai/gym/tree/0.15.4#rendering-on-a-server
xvfb-run -s "-screen 0 1400x900x24" bash
```
