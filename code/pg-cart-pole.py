import gym
import numpy as np
from collections import defaultdict
import time

from wondevwoman import weight_logger, weight_plotter

env = gym.make('CartPole-v0')

batch_size = 1
rms_decay_rate = 0.99
learning_rate = 0.0005

# dim of observation vector (input)
i = 4
# number of hidden neurons
h = 100 #20
# dim of action probability vector (output)
o = 1
# init weights
W1 = np.random.randn(h, i) / np.sqrt(i)
W2 = np.random.randn(o, h) / np.sqrt(h)
# gradient buffers for accumulating the batch
dW1 = np.zeros_like(W1)
dW2 = np.zeros_like(W2)
# rmsprop buffer (gradient descent optimizer)
rmspropW1 = np.zeros_like(W1)
rmspropW2 = np.zeros_like(W2)
# value storage to be filled during a match
history = defaultdict(list)

log_filename = 'pg-cart-pole.log'
log = weight_logger.WeightLogger(log_filename, overwrite=True)


def sigmoid(x):
    '''an activation function'''
    return 1. / (1. + np.exp(-x))

def d_sigmoid(x):
    '''its derivative'''
    return np.exp(x) / ((np.exp(x) + 1.) ** 2.)

def rect(x):
    '''rectifier (ReLU) activation function'''
    x[x < 0] *= 0 #.001
    return x

def d_rect(x):
    '''its derivative'''
    negatives = x < 0
    x[negatives] = 0 #.001
    x[np.logical_not(negatives)] = 1
    return x

def discount_rewards():
    gamma = 0.5 # reward back-through-time aggregation ratio
    r = np.array(history['r'], dtype=np.float64)

    if np.all(r == 1):
        return None

    for i in range(len(r) - 2, -1, -1):
        #r[i] = gamma * r[i] + (1 - gamma) * r[i+1]
        r[i] = r[i] + gamma * r[i+1]
    return r

def net_forward(x):
    '''forward pass through the net. returns action probability vector and hidden state.'''
    # hidden layer
    h = np.dot(W1, x)
    # rectifier function (ReLU)
    h = rect(h)
    # output layer
    p = np.dot(W2, h)
    # sigmoid activation function
    p = sigmoid(p)
    return p, h

def net_backward(rewards):
    '''does backpropagation with the accumulated value history and processed rewards and returns net gradients'''

    dW1, dW2 = np.zeros_like(W1), np.zeros_like(W2)

    # iterate over history of observations, hiddens states, probabilities, delta-probabilities and discounted rewards
    for x, h, p, dp, r in zip(history['x'], history['h'], history['p'], history['dp'], rewards):
        dpr = r * dp * d_sigmoid(p)
        dW2 += np.dot(dpr[:, np.newaxis], h[:, np.newaxis].T)
        dh = dpr.dot(W2)
        dh *= d_rect(h) # chain rule: set dh (outer) to 0 where h (inner) is <= 0
        dW1 += np.dot(x[:, np.newaxis], dh[:, np.newaxis].T).T

    return dW1, dW2


try:
    nb_games = 0
    max_steps = 0
    avg_nb_steps = None
    while True:
        observation = env.reset()
        done = False
        nb_steps = 1
        while not done:
            history['x'].append(observation)
            p, h = net_forward(observation)
            history['h'].append(h)
            history['p'].append(p)
            action = 0 if p < np.random.random() else 1
            history['dp'].append(action - p)

            observation, reward, done, info = env.step(action)
            nb_steps += 1

            history['r'].append(-1 if done and nb_steps < 201 else 1) # `reward` is always 1

        nb_games += 1
        if avg_nb_steps is None:
            avg_nb_steps = nb_steps
        else:
            avg_nb_steps = 0.99 * avg_nb_steps + 0.01 * nb_steps
        max_steps = max(max_steps, nb_steps)

        r = discount_rewards()

        if r is not None:
            r -= np.mean(r)
            std = np.std(r)
            r /= std
            # get gradients with backprop and pimped rewards
            dW1, dW2 = net_backward(r)
            # aggregate gradients for later use
            dW1 += dW1
            dW2 += dW2
        history.clear()

        if nb_games % batch_size == 0:
            # time to learn
            # do fancy rmsprop-optimized gradient descent
            rmspropW1 = rms_decay_rate * rmspropW1 + (1 - rms_decay_rate) * dW1**2
            W1 += learning_rate * dW1 / (np.sqrt(rmspropW1) + 0.00001)
            # clear gradient buffer
            dW1 = np.zeros_like(dW1)
            # and second layer
            rmspropW2 = rms_decay_rate * rmspropW2 + (1 - rms_decay_rate) * dW2**2
            W2 += learning_rate * dW2 / (np.sqrt(rmspropW2) + 0.00001)
            dW2 = np.zeros_like(dW2)

            log.log((W1, W2))

            print('\rgame #%i avg number of steps: %.2f (max: %i)' % (nb_games, avg_nb_steps, max_steps), end='', flush=True)

except KeyboardInterrupt:
    print('\nTrained %i games. Showtime!' % nb_games)

    observation = env.reset()
    done = False
    while not done:
        env.render()

        p, h = net_forward(observation)
        action = 0 if p < np.random.random() else 1
        observation, reward, done, info = env.step(action)
        time.sleep(0.075)
