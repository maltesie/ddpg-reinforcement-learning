# Robotics Project - Model Based RL

## Goal:

We want to compare model based RL methods with model free ones on different tasks and develop a system of 
attributes to distinguish between between them. This could lead to a better understanding of where to use model
based RL.

Some Pros and Cons to maybe produce the system of atrributes:


Pro model based:

* learns well from examples, demonstrations
* more data efficient


Contra model based:

* small model errors can have great effect on policy
* we have to make assumptions of how the world works to model it


Pro model free:

* direct learning of the best policy
* no assumptions


Contra model free:

* less data efficient


## Environments:


### OpenAI Gym

We use the master branch of the git repository:

https://github.com/openai/gym.git


## Methods:


### Guided Policy Search:

We use the code provided in this git repository:

https://github.com/cbfinn/gps.git

A detailed instruction with examples can be found here:

http://rll.berkeley.edu/gps/


### Relative Entropy Policy Search:

Implementation of the method in python:

https://github.com/rll/rllab/blob/master/rllab/algos/reps.py

from the rllab framework:

https://github.com/rll/rllab


### Neural Fitted Q:

Implementation of the method using dropout regularization and convolutional neural networks:

https://github.com/cosmoharrigan/rc-nfq.git


### Deep Deterministic Policy Gradients:

We use the following implementation of the method for OpenAI Gym:

https://github.com/stevenpjg/ddpg-aigym.git
