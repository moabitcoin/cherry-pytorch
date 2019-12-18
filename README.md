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
# If on a headless machine of
vizdoom.vizdoom.ViZDoomErrorException: Could not initialize SDL video:
No available video device
export DISPLAY=:0
```

## Arcade Learning Environment (ALE)
```
https://github.com/openai/gym/blob/master/docs/environments.md#atari
```

## OpenAI RL baselines
Needed for Deepmind styled pre-processing wrappers. Needs TF>=1.14 :woman_facepalming:
```
pip install tensorflow_gpu==1.14
https://github.com/openai/baselines/tree/master#installation
```

## Remote Render
```
https://github.com/openai/gym/tree/0.15.4#rendering-on-a-server
xvfb-run -s "-screen 0 1400x900x24" bash
```
