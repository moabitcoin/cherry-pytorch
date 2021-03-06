## Modular

The core principle of `cherry` is to organise an RL framework around disjoint portable module. These portable modules represent the Agent/Environment de-coupling presented in [spinning up.](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#id4) We compose the Agent as its function approximation (Deep NN) representation. You can check out our [Agents](https://github.com/moabitcoin/cherry-pytorch/tree/master/cherry/agents) and [Environment](https://github.com/moabitcoin/cherry-pytorch/tree/master/cherry/envs)

<p align="center">
<img src="https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png">
</p>

### Agent
The Agent API is structured as follows

```
class AgentName():

  def __init__(self, cfgs, model=None, model_file=None,
               device=None, log_level='info'):

  @property
  def state(self):

  @state.setter:
  def state(self)

  @state.getter:
  def state(self)

  @property
  def reward(self):

  @reward.setter
  def reward(self):

  @reward.getter
  def reward(self):

  @property
  def ep_reward(self):

  @ep_reward.setter
  def ep_reward(self):

  @ep_reward.getter
  def ep_reward(self):

  def reset(self):

  def flash(self):

  def init(self, model_file):

  def augment_state(self):

  def append_state(self, state):

  def action(self, state, deterministic=False):

  def train(self, env, train_cfgs, gitsha, model_dest):

  def play(self, env, test_cfgs, gitsha):

  def optimize(self):

```
### Architecture
The Architecture API is structures as follows

```
class ModelName(torch.nn.Module):

  def __init__(self, input_shape, state_size, action_size, device):

    super(ModelName, self).__init__()

    self.device = device
    self.state_size = state_size
    self.action_size = action_size
    self.input_shape = input_shape

  def init_weights(self, m):

  def forward(self, x):

```
### Environment
The Environment API is structures as follows

```
class EnvironmenName():

  def __init__(self, cfgs, play=False):

    self.env = None
    self.seed = cfgs.get('seed')
    self.env_name = cfgs.get('name')
    self.env_solution = cfgs.get('env_solution')

  def reset(self):

  def step(self, action):

  def close(self):

  def update_env(self, update_fn, **kwargs):

```

## Reproducibility
### Pseudorandom number gen
Taking a cue out of [PyTorch](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility) and our sanity we set the [`random`](https://pytorch.org/docs/stable/notes/randomness.html#numpy) seeds from `yaml` config files as follows. 
```
  env = make_atari(self.env_name)
  self.env = wrap_deepmind(env)
  self.env.seed(self.seed)
  torch.manual_seed(self.seed)
```
If you are using random number generator from within numpy in your codes. Please do set the same random seed following the PyTorch [convention](https://pytorch.org/docs/stable/notes/randomness.html#numpy). 
### Tracking code & models
Training/testing is setup through `yaml` config files. Sample files located [here](https://github.com/moabitcoin/cherry-pytorch/blob/master/configs/control.yaml). During training, the config file & model weights are saved at `model_dest` and appended with [`commit-gitsha`](https://gist.github.com/masak/2415865) of the repo for traceability & reproducibility. We follow the PyTorch Hub's convention of using [first 8 chars](https://github.com/pytorch/pytorch/blob/master/torch/hub.py#L459) of commit gitsha as for tagging files & models. F.ex

```
# commit gitsha
git log -n 1
commit c099dff57dac5dbd4bccce7cfb5f8e98c145ee2d (tag: v1.0)
Author: Freja <xxx@yyy.com>
Date:   Tue Jan 28 14:26:03 2020 +0100

    readme update
```
```
# <model_dest> from configs/control-ddpg.yaml
ls <model_dest>
-rw-rw-r-- 1 moabit 71425 Jan 27 16:14 agent-000030000-c099df.pth
-rw-rw-r-- 1 moabit 71425 Jan 27 16:15 agent-final-c099df.pth
-rw-rw-r-- 1 moabit  1674 Jan 27 16:12 control-ddpg-c099df.yaml
```
