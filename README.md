<h1 align='center'>Cherry in PyTorch</h1>

Reinforcement Learning &amp; other bedtime stories in PyTorch. This repository is [pytorch](https://pytorch.org) implementation of the [Intro RL course](https://github.com/simoninithomas/Deep_reinforcement_learning_Course) & [Advanced RL course](https://github.com/dennybritz/reinforcement-learning). The name cherry is derived from Yann Lecun's claim of RL being the cherry on the ML cake. Before training/playing the agent please install the external [dependencies](https://github.com/moabitcoin/cherry-pytorch#dependencies)

## :computer: Installation
### :snake: Conda
```
conda create -n cherry python=3.6
conda activate cherry
pip install -r requirements.txt
pip install -e .
```
### :whale: Docker

## :tada: Usage
```
# Inside the virtual environment
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
# Training
cherry train -c configs/control.yaml -d cpu
# Playing
cherry play -c configs/control.yaml -d cpu
```

## Dependencies
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
Needed for Deepmind styled pre-processing wrappers. Needs TF>=1.14 :woman_facepalming:
```
pip install tensorflow_gpu==1.14
https://github.com/openai/baselines/tree/master#installation
```

## :tv: Remote Render
```
https://github.com/openai/gym/tree/0.15.4#rendering-on-a-server
xvfb-run -s "-screen 0 1400x900x24" bash
```
