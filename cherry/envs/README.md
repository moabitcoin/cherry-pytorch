# Classic Control
We wrap the classic controls problems first introduced by Richard Sutton into the domain of RL. These are ported as is from [OpenAI's GYM.](https://gym.openai.com/envs/#classic_control)

# Arcade Learning Environment (ALE)
We wrap the Arcade Learning Environment (ALE) which includes some of the classic [Atari 2600 games.](https://gym.openai.com/envs/#atari). We futher wrap the GYM env to include few prosed changes to default environment from Deepmind outlined [here](https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py#L275)

# VizDoom
[Doom](http://vizdoom.cs.put.edu.pl) is classic FPS game form the the early 90s. Is available both in single and multi-agent environment. The authors provided an easy to use pythonic interface for RL commiunity to use. We wrap vizdoom with [minimal](https://github.com/moabitcoin/cherry-pytorch/blob/docs/cherry/envs/doom.py#L17) changes.
