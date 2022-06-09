# Q-LEARNING WITH TREASURE MAP GAME

## Introduction

In this repository I present my struggles with the implementation of the Q-learning algorithm for an easy game in which the treasure hunter has to avoiding pirates to find buried treasure.

On a board of a certain size (in this example 10x10) there are:
- pirates - obstacles to avoid (yellow)
- buried treasure - target to be found (navy blue)

Board example:<br>
![exmpl_treasure_map](/img/exmpl_map.png)

The goal of the game is **to get to the hidden treasure**. Both the starting point and the end point are selected randomly (although there is possibility to define environment manually).

Our goal is to use the Q-learning algorithm to teach agent playing the above game.

First, let's describe the basic assumptions: <br>
We assume that the agent sees only the fields surrounding him.<br>
![agent_view](/img/agent_view.png)<br>
To prevent the agent from falling off the board, we have defined a method that in the case of illegal movement causes the agent to move in the opposite direction - agent bounces off the wall.


What is the **environment**?
- state - information about the state of the environment. In our case it is a tuple (-9, 0, 1, -9, 0, -9, 1, -1) corresponding to our agent's field of view. Each value corresponds to the type of the specified field:
    - -9 (off the board),
    - 0 (available),
    - 1 (pirate),
    - -1 (treasure).
- epizode - a set of steps after which the state of the environment is reset. In our case, one game, i.e. all agnet steps until he finds the treasure.
- reward - information returned by the environment after performing a step, which should be interpreted as a reward or a penalty for the action. For example:
    - -50 if we come across a pirate
    - 1 if we enter the available field (not pirate, not treasure)
    - 100 if we find the treasure
    <br>
**Action** basically it is a agent movement to the next field.<br>

Who is the **agent**?<br>
An agent is an element that interacts with the environment. Its task is to maximize the reward, that is, to learn the most beneficial strategy of interaction with the environment. In our case, it is an adventurer moving across the map.

**Policy** - is a function that takes an observation as input and returns an action. This is usually the probability distribution on actions available in the certain state. It can be said that this is a kind of instruction for a treasure hunter on how to move around the map. Good policy should say: avoid pirates, go to the treasure.

**In short, the agent's task is to find the policy, which maximizes the expected value of the sum of rewards:**<br>
![equation_expected_policy](/img/equation_expected_policy.png)<br>

How will an agent learn to play the game?<br>
The simplest form of Q-learning is one-step Q-learning, defined by:<br>
![q_learning_equation](/img/q_learning_equation.png)<br>
In this case, the learned action-value function, Q, directly approximates q∗, the optimal action-value function, independent of the policy being followed. In each step, the policy values are updated with the reward for the action taken and the best possible next move.<br>
In the treasure hunter's language, the Q-learning algorithm is based on updating instructions on how to move by moving around the map and learning from mistakes.<br>



## Experiments
I have experienced first-hand that training an agent with reinforcement learning methods is not so easy. Learning process in RL is much more difficult than supervised learning training. Later I will describe some of the difficulties I encountered. Now I will present some examples of the results that I was able to get.<br>
Let's assume that we teach the agent all the time in the same environment, i.e. all training takes place on the same map (generalization to other maps is not so simple).<br>
![exmpl_map](/img/exmpl_map.png)<br>

Next We will describe the applied hyperparameters:
- n_iter = 10000 - number of epizodes
- alpha = 0.1 (learning rate) - the amount that the policy values are updated during training
- gamma = 0.8 (discount factor) - determines the importance of future rewards
- epsilon = (n_iter-1)*0.6/n_iter - determines the probability of a random movement. Changes over time and goes down to zero.

How to check if the agent is learning at all or if it is not making even bigger mistakes than at the beginning? How do we know if learning is proceeding correctly?<br>
One way is to measure the sum of the rewards in each episode (in time).<br>
The learning history was as follows:<br>
![learning_history](/img/learning_history.png)<br>
The sum of the rewards should increase over time.<br>

Let's check what our agent has learned:<br>
![agent_show](/img/agent_show.png)<br>
We can see in which situations (states in the environment), where (in which direction) the agent should go.<br>

Once the agent has been trained, we can check how he is doing on the map he was learning on:<br>
![trained_agent](/img/trained_agent.gif)<br>

Unfortunately, on the other map, agent does not do so well anymore. He often gets into loops, like this time:<br>
![exmpl_stuck](/img/exmpl_stuck.gif)<br>


## Problems and struggles
1. Initially, the field of view was defined only in four directions (N, S, W, E). Unfortunately, it turned out to be too poor field of view. Most of time, agent got stuck at local minima, which made him circulate in an endless loop.
2. The learning process is closely related to rewarding. Rewards should be carefully defined. In simple environments like this, it's not that hard. In more complex environments it is probably more difficult to ensure that the agent learns "right". On the other hand, by changing the reward, we can observe whether, for example, the agent is able to pass through the pirate in certain cases to leave the closed area.
3. The biggest problem was dealing with getting stuck at local minima. First, I removed the "small" reward for visiting an empty field (no pirate). However, that did not solve the problem. It turned out to be better for me to save the history of the path traveled and punished for re-entering the field once visited.
4. In the beginning, the dictionary of states and the policy matrix were initialized entirely (with states of length 4, it was enough). However, when increasing the agent's field of view, the size of the dictionary and matrix increased exponentially, making it impossible to train in a finite time. At some point I decided to initiate only the initial state and add it to the dictionary with each new visited state.
5. Much more, but I don't remember :)

<br>
<br>

The inspiration for the above experiments was the book:<br>
_Reinforcement Learning: An Introduction - Richard S. Sutton and Andrew G. Barto_
