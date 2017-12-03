import gym
import numpy as np
from collections import defaultdict
import time

from wondevwoman import weight_logger, weight_plotter

env = gym.make('CartPole-v1')

batch_size = 1
rms_decay_rate = 0.99
learning_rate = 0.000333

# dim of observation vector (input)
i = 4
# number of hidden neurons
h1 = 100
h2 = 100
# dim of action probability vector (output)
o = 1 + i # 1 action + model
# init weights
W1 = np.random.randn(h1, i) / np.sqrt(i)
W2 = np.random.randn(h2, h1) / np.sqrt(h1)
W3 = np.random.randn(o, h2) / np.sqrt(h2)

# gradient buffers for accumulating the batch
dW1 = np.zeros_like(W1)
dW2 = np.zeros_like(W2)
dW3 = np.zeros_like(W3)
# rmsprop buffer (gradient descent optimizer)
rmspropW1 = np.zeros_like(W1)
rmspropW2 = np.zeros_like(W2)
rmspropW3 = np.zeros_like(W3)
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
    '''leaky rectifier (ReLU) activation function'''
    x[x < 0] *= 0.1
    return x

def d_rect(x):
    '''its derivative'''
    negatives = x < 0
    x[:] = 1
    x[negatives] = 0.1
    return x

def get_normalized_state(x):
    '''shifts and scales the 0..1 net output vector to match the state bounds'''
    y = np.zeros_like(x)
    y[0] = (x[0] + 2.4) / 4.8
    y[1] = (x[1] + 4.) / 8.
    y[2] = (x[2] + 0.20943951) / 0.41887902
    y[3] = (x[3] + 4.) / 8.
    return y

def discount_rewards():
    gamma = 0.99 # reward back-through-time aggregation ratio
    r = np.array(history['r'], dtype=np.float64)

    if np.all(r == 1):
        return None

    for i in range(len(r) - 2, -1, -1):
        r[i] = r[i] + gamma * r[i+1]
    return r

def net_forward(x):
    '''forward pass through the net. returns action probability vector and hidden state.'''
    # hidden layer 1
    h1 = np.dot(W1, x)
    h1 = rect(h1)
    # hidden layer 2
    h2 = np.dot(W2, h1)
    h2 = rect(h2)
    # output layer
    p = np.dot(W3, h2)
    p = sigmoid(p)
    return p, h1, h2

def net_backward(rewards):
    '''does backpropagation with the accumulated value history and processed rewards and returns net gradients'''

    dW1, dW2, dW3 = np.zeros_like(W1), np.zeros_like(W2), np.zeros_like(W3)

    # iterate over history of observations, hiddens states, net outputs, actions taken, rewards and next observations
    for x, h1, h2, p, a, r, xn in zip(history['x'], history['h1'], history['h2'], history['p'], history['a'], rewards, history['xn']):
        # action gradient and next observation estimation error
        dp = np.hstack(([r * (a - p[0])], get_normalized_state(xn) - p[1:])) * d_sigmoid(p)
        #dp = r * (a - p[0]) * d_sigmoid(p) # model-free variant
        dW3 += np.dot(dp[:, np.newaxis], h2[:, np.newaxis].T)

        dh2 = dp.dot(W3)
        dh2 *= d_rect(h2)
        dW2 += np.dot(dh2[:, np.newaxis], h1[:, np.newaxis].T)

        dh1 = dh2.dot(W2)
        dh1 *= d_rect(h1) # chain rule: set dh (outer) to 0 where h (inner) is <= 0
        dW1 += np.dot(x[:, np.newaxis], dh1[:, np.newaxis].T).T

    return dW1, dW2, dW3


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
            p, h1, h2 = net_forward(observation)
            history['h1'].append(h1)
            history['h2'].append(h2)
            history['p'].append(p)
            action = 0 if p[0] < np.random.random() else 1
            history['a'].append(action)

            observation, reward, done, info = env.step(action)
            nb_steps += 1

            history['r'].append(-1 if done and nb_steps < 501 else 1) # `reward` is always 1
            history['xn'].append(observation)

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
            ddW1, ddW2, ddW3 = net_backward(r)
            # aggregate gradients for later use
            dW1 += ddW1
            dW2 += ddW2
            dW3 += ddW3
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

            rmspropW3 = rms_decay_rate * rmspropW3 + (1 - rms_decay_rate) * dW3**2
            W3 += learning_rate * dW3 / (np.sqrt(rmspropW3) + 0.00001)
            dW3 = np.zeros_like(dW3)

            log.log((W1, W2, W3))

            print('\rgame #%i avg number of steps: %.2f (max: %i)' % (nb_games, avg_nb_steps, max_steps), end='', flush=True)

except KeyboardInterrupt:
    print('\nTrained %i games. Showtime!' % nb_games)

    observation = env.reset()
    done = False
    while not done:
        env.render()

        p, _, _ = net_forward(observation)
        action = 0 if p[0] < np.random.random() else 1
        observation, reward, done, info = env.step(action)
        time.sleep(0.075)
