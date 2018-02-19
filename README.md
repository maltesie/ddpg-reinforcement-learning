# Robotics Project - Model Based RL

We implemented model based RL with REINFORCE and DDPG. All code is implemented in Python 3.

## DDPG

The code is located in the `ddpg` folder.

### Requirements

A working Python 3 installation or virtual environment is reqired. It has to have following site-packages installed, which are all in the pip-repository:

* tensorflow
* sklearn
* numpy
* scipy
* matplotlib
* gym

### Set Parameters

All parameters for the training can be set in the `ddpg_with_model.py` file. Following parameters can be set:

* In the meta parameter block you can set a session name and set the number of trainings to be run
* In the model parameter block one can toggle the use of a model, the model type and model pretraining conditions.
* In the training parameter block the exploration noise and training length can be set.
* In the utility parameter block the environment can be chosen.

The code in its current form can only run in the environments CartPole-v0, CartPole-v1 and Pendulum-v0 with a model.

### Run

To run the training(s) with the set parameters, just execute the file `ddpg_with_model.py`.

### Plot

Plots of the data can be made with the file `plot_results.py`. The file has to be adjusted to the session name and the amount of trainings.

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

To recreate the results of the DDPG part of this project one has to run the file `ddpg_with_model.py` 10 times.
Every time a new session name has to be set. Here are the parameter combinations that lead to the resulting data of this project:

* use_model = False, ENV_NAME = 'Pendulum-v0'
* use_model = False, ENV_NAME = 'Cartpole-v1'
* use_model = True, ENV_NAME = 'Pendulum-v0', M = GP, nb_ep_eval = 5
* use_model = True, ENV_NAME = 'Pendulum-v0', M = GP, nb_ep_eval = 10
* use_model = True, ENV_NAME = 'Pendulum-v0', M = NN, nb_ep_eval = 5
* use_model = True, ENV_NAME = 'Pendulum-v0', M = NN, nb_ep_eval = 10
* use_model = True, ENV_NAME = 'Cartpole-v1', M = GP, nb_ep_eval = 5
* use_model = True, ENV_NAME = 'Cartpole-v1', M = GP, nb_ep_eval = 10
* use_model = True, ENV_NAME = 'Cartpole-v1', M = NN, nb_ep_eval = 5
* use_model = True, ENV_NAME = 'Cartpole-v1', M = NN, nb_ep_eval = 10

## REINFORCE

The code is located in the `pg` folder.

### Requirements

Python 3:
* tensorflow
* numpy
* matplotlib
* gym

### Usage

For a demo, run `launch_cartpole.py demo`.

To evaluate a parameter set, define it in the launch file at `param = { ... }` and run `launch_cartpole.py evaluate`.
This will create a file called `eval-scores` following the parameters ending with `.txt` containing the (hopefully increasing) rewards of one training run per line in a list.
You can plot these files with `plot_eval.py <files> 100` assuming you did not change the batch size.

To evaluate the model, you first need to create a train/test set of cartpole trajectories with `gen_trajectories.py [samples_per_dimension [filename]]`.
`samples_per_dimension` defines the number of equally sized divisions in which the interval of each dimension is split into and defaults to 4, resulting in 256 trajectories.
`filename` defaults to `cartpole-trajectories.txt`.
Then run `train_model.py [filename]` to train the model with different parameter sets and plot the evaluation on the test set.
You can change the parameter sets in the file as you wish.

### Interesting Parameters

The agent parameters default to the values stated in the `Agent` constructor in `tf_cart_pole_agent.py`.
They can be overwritten in `launch_cartpole.py` or `train_model.py` respectively.

**`learning_rate`**: Directly handed over to a `tf.Optimizer`, shouldn't needed to be changed. But note that model learning rate is `learning_rate / 10`.

nb_world_features: Controlls the neural net width. Model hiddel layer width is `nb_world_features * 10`.

**`rect_leakiness`**: Controlls leaky ReLU leakiness.

**`learn_model`**: Which model to use (none/delta/absolute).

**`sample_model`**: Whether to use the learned model (legancy reasons, sorry).

**`model_training_noise`** and **`model_training_noise_decay`**: Controlls model training noise ratio and decay respectively. Refer to our paper if unclear.

**`model_afunc`**: Activation function of the model net's hidden layer.

**`replay_buffer_size`**: Target number of samples in the model experience replay buffer. The buffer gets filtered down wrt. model error as soon as it exceeds `replay_buffer_size * 1.5` entries.

**`multitask`**: Which model to use for multitask-learning the policy (none/delta/absolute).

**`random_seed`**: Tensorflow seed to controll weight initialization.