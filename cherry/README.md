## Modular

The core principle of `cherry` is to organise an RL framework around disjoint portable module. These portable modules represent the Agent/Environment de-coupling presented in [spinning up.](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html#id4) We compose the Agent as its function approximation (Deep NN) representation. You can check out our [Agents](https://github.com/moabitcoin/cherry-pytorch/tree/master/cherry/agents) and [Environment](https://github.com/moabitcoin/cherry-pytorch/tree/master/cherry/envs)

<p align="center"> 
<img src="https://spinningup.openai.com/en/latest/_images/rl_diagram_transparent_bg.png">
</p>

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
