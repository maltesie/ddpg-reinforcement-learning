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