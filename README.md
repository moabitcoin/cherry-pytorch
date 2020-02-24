<h1 align='center'>:cherries: Cherry in PyTorch</h1>

Reinforcement Learning &amp; other bedtime stories in PyTorch. This repository is [pytorch](https://pytorch.org) implementation of the [Intro RL course](https://github.com/simoninithomas/Deep_reinforcement_learning_Course) & [Advanced RL course](https://github.com/dennybritz/reinforcement-learning). The name `cherry` is derived from Yann Lecun's [statement](https://www.quora.com/What-do-you-think-about-reinforcement-learning-Is-it-the-cherry-on-the-cake-as-Yann-LeCun-puts-it-1) regarding RL being the cherry on the ML [cake.](https://miro.medium.com/max/1200/1*bvMhd_xpVxfJYoKXYp5hug.png) For training/playing the agent within [conda](#snake-conda) env, please install the [external dependencies](https://github.com/moabitcoin/cherry-pytorch#two_men_holding_hands-dependencies). Not to be confused with a package of [similar name.](http://cherry-rl.net/tutorials/getting_started/)

# Table of Contents
* [Installation](#computer-installation)
  - [Linux](#penguin-linux)/[macOS](#green_apple-macos) or [Docker](#whale-docker)
* [Sample usage](#tada-usage)
  - [Cartpole-v0](#eyes-example-cartpole)
* [Dependencies (Conda only)](#two_men_holding_hands-dependencies)
  - [Vizdoom](#godmode-vizdoom), [ALE](#space_invader-arcade-learning-environment), [Baselines](#chart_with_upwards_trend-openai-rl-baselines)
* [Architectures](#architectures) & [Agents](#agents)
* [Environments](#environments)
* [Design principle](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/README.md)
* [Reproducibility](https://github.com/moabitcoin/cherry-pytorch/tree/master/cherry#reproducibility)

## :computer: Installation
```
Tested with ubuntu:18.04/macOS-Mojave:10.14.6
```

### :penguin: Linux
```
make linux
```

### :green_apple: MacOS
```
make mac
```

### :whale: Docker
```
# CPU
make install dockerfile=Dockerfile.cpu dockerimage=moabitcoin/cherry-pytorch-cpu
make run dockerimage=moabitcoin/cherry-pytorch-cpu
# GPU
make install dockerfile=Dockerfile.gpu dockerimage=moabitcoin/cherry-pytorch-gpu
make gpu dockerimage=moabitcoin/cherry-pytorch-gpu
```

## :tada: Usage
```
# Inside the virtual environment / Docker
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
### :eyes: Example (Cartpole)
#### Train
```
cherry train -c configs/control.yaml -d cpu
```
#### Play
```
# <model_dest> in configs/control.yaml
cherry play -c <model_dest>/control-<commit-gitsha>.yaml -d cpu -m <model_dest>/agent-final-<commit-gitsha>.pth
```
#### Visualise
```
# <state_dest> in configs/control.yaml
vlc <state_dest>
```

## Architectures
We support [4 feedforward](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/README.md#architectures) architectures within `cherry`. We plan to expand the list of architectures to include Recurrent/Transformer/Memory architectures. If your personal model flavour is missing, please open [an issue](https://github.com/moabitcoin/cherry-pytorch/issues) with links to architecture details.

## Agents
We support [4 Agents](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/README.md) within `cherry`. We plan to expand the list of agents to include TRPO/PPO. Please feel free to make an agent request by opening [an issue](https://github.com/moabitcoin/cherry-pytorch/issues) with useful links to publication(s)/existing implementation.

## Environments
We support [4 environments](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/envs/README.md) within `cherry`. This list is planned to be expanded to include [Robotics](https://gym.openai.com/envs/#robotics) and other [3rd party](https://github.com/openai/gym/blob/master/master/environments.md#third-party-environments) environments.

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
