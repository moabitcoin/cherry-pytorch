# Classic Control
We wrap the classic controls problems first introduced by Richard Sutton into the domain of RL. These are ported as is from [OpenAI's GYM.](https://gym.openai.com/envs/#classic_control) A nice `hello world` on-boarding to `cherry` can be performed by solving  `CartPole-v0` with `VPG` outlined [here.](https://github.com/moabitcoin/cherry-pytorch/blob/master/configs/control.yaml)

# Arcade Learning Environment (ALE)
We wrap the Arcade Learning Environment (ALE) which includes some of the classic [Atari 2600 games.](https://gym.openai.com/envs/#atari). We futher wrap the GYM env to include few prosed changes to default environment from Deepmind outlined [here](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L275). These changes include
- End of life := [End of episode](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L61)
- Reward [clipping](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L125) to [-1, +1]
- [Fire reset](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L41) at End of life
- Frame [Skip](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L97)

# VizDoom
[Doom](http://vizdoom.cs.put.edu.pl) is classic FPS game form the the early 90s. Is available both in single and multi-agent environment. The authors provided an easy to use pythonic interface for RL commiunity to use. We wrap vizdoom with [minimal](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/envs/doom.py#L17) changes. We include the few [scenarios](https://github.com/moabitcoin/cherry-pytorch/tree/docs/cherry/envs/assets) which are popular with RL community to try. Including the Hello-wordish [basic](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/envs/assets/basic.cfg) scenario.
- `Basic` : Single target at the back of the wall
- `Death Match` : Multi agent death match
- `Health Gathering` : Learn to survive by gather med-packs
- `Deadly Corridor` : Learn to navigate a maze & survive by terminating zombies
