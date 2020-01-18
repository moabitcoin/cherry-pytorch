# Agents
## DQN
[DQN](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning) is a seminal work from Deepmind which brought limelight back to (Deep) Re-reinforcement learning. DQN was applied to play [Atari](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning) and later Doom. We ported many ideas from PyTorch's [DQN example](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html).

Key ingredients for DQN are
- [Replay buffer](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L6) and [replay buffer size](https://github.com/moabitcoin/cherry-pytorch/blob/master/configs/doom-dqn.yaml#L36)
- Decoupled [Policy and Target](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/dqn.py#L58) models for stability
- Policy [update](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/dqn.py#L252) & Target [update](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/dqn.py#L255)

## DDQN
[Double DQN](https://arxiv.org/abs/1509.06461) aimed at improving one of the shortcomings of DQN. Specifically the over-estimation of action value function. [This](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/agents/ddqn.py#L169) improves training stability and in some of the Atari 2600 games improves model performance. DDQN uses same ingredients as DQN above.

## VPG
Vanilla Policy Gradient is an on-policy method for training an agent. Unlike DQN/DDQN which are off policy methods. OpenAI's [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/vpg.html) has a great tutorial explaining it in easy to digest form.

Key ingredients for VPG are
- [Policy rollout(s)](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/agents/vpg.py#L289) : Batch consisting of multiple trajectories
- [Discounting rewards](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/agents/vpg.py#L173) to go
- De-couple [Action Value](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/agents/vpg.py#L225) & [Value function](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/agents/vpg.py#L239)
- Negative Log-likelihood of [`Q(s, a)`](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/agents/vpg.py#L231)

## A2C

# Architectures
## [MLP](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L208)
- 1 Linear layer
- [2 Heads](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L208)
  - [Action Value](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L223) function
  - [Value function](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L224)

## [Convnet Small](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L50)
- 2 Conv layers
- 1 Linear layer
- [Two heads](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L50) :
    - [Action value](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L98) function (Q)
    - [Value function](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L99) (V)

## [Convnet Medium](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L105)
- 3 Conv layers + Batch Norm
- [Two heads](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L113) :
    - [Action value](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L134) function (Q)
    - [Value function](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L135) (V)

## [Convnet Large](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L160)
- 3 Conv layers
- 1 Linear layer
- [Two heads](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L160) :
    - [Action value](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L179) function (Q)
    - [Value function](https://github.com/moabitcoin/cherry-pytorch/blob/master/cherry/agents/models.py#L180)(V)
