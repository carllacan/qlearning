# Q-learning
Using reinforcement learning to play snake and similar games.

An environment is created that simulates a game . This environment takes actions and returns the resulting screen of those actions, plus the resulting reward (-1 if the player dies, 0 if nothing happens and 1 if the player scores).

A player with a neural-network provides actions and learns from the environment responses using Q-learning per advantage learning.

The network should learn what actions provide the best value. So far it has had mild success... in the best run the player has managed to survive for around 5 turns consistently.