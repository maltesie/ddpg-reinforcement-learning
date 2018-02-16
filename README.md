# Robotics Project - Model Based RL

We implemented model based RL with REINFORCE and DDPG. All code is implemented in Python 3.

## REINFORCE

### Requirements

### Usage

## DDPG

The code is located in the ddpg folder. 

### Requirements

A working Python 3 installation or virtual environment is reqired. It has to have followeing site-packages installed, which are all in the pip-repository:

* tensorflow
* sklearn
* numpy
* scipy
* matplotlib
* gym

### Set Parameters

All parameters for the training can be set in the ddpg_with_model.py file. Following parameters can be set:

* In the meta parameter block you can set a session name and set the number of trainings to be run
* In the model parameter block one can toggle the use of a model, the model type and model pretraining conditions. 
* In the training parameter block the exploration noise and training length can be set. 
* In the utility parameter block the environment can be chosen. 

The code in its current form can only run in the environments CartPole-v0, CartPole-v1 and Pendulum-v0 with a model.

### Run

To run the training(s) with the set parameters, just execute the file ddpg_with_model.py.

### Plot

Plots of the data can be made with the file plot_results.py. The file has to be adjusted to the session name and the amount of trainings. 

The current file uses the projects results and recreates the graphics used in the presentation and report. The wanted configuration has to be uncommented.

### Disclaimer

The code in the folder ddpg is based on this implementation of DDPG:

https://github.com/liampetti/DDPG

The code was changed and improved in those files:

* ddpg_with_model.py

The code was kept as the original in those files:

* noise.py
* ddpg.py
* replaybuffer.py
* reward.py
* actor.py
* critic.py

The code in the  following files was written for this project by Malte Siemers:

* plot_results.py
* actionsampler.py
* functions.py
* nn.py
* gp.py

### Recreate Results

To recreate the results of the DDPG part of this project one has to run the file ddpg_with_model.py 10 times. 
Every time a new session name has to be set. Here are the parameter combinations that lead to the resulting data of this project:

* use_model = False, ENV_NAME = 'Pendulum-v0'
* use_model = False, ENV_NAME = 'Cartpole-v1'
* use_model = True,
* use_model = True,
* use_model = True,
* use_model = True,
* use_model = True,
* use_model = True,
* use_model = True,