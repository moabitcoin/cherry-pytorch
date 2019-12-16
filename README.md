<h1 align='center'>Cherry in PyTorch</h1>
Reinforcement Learning Tutorials &amp; other bedtime stories in PyTorch

This repositories is pytorch implementation of the [Intro RL course](https://github.com/simoninithomas/Deep_reinforcement_learning_Course) & [Advanced RL course](https://github.com/dennybritz/reinforcement-learning)

## Base Installation

```
conda create -n cherry
conda install pip
pip install -r requirements.txt
pip install -e .
```

## ViZDoom
Install dependencies as outlined [here](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps)
```
sudo apt-get install libcanberra-gtk-module
pip install vizdoom
# If on a headless machine
export DISPLAY=:0
```

## Arcade Learning Environment (ALE)
```
https://github.com/openai/gym/blob/master/docs/environments.md#atari
```


## Remote Render
```
https://github.com/openai/gym/tree/0.15.4#rendering-on-a-server
xvfb-run -s "-screen 0 1400x900x24" bash
```
