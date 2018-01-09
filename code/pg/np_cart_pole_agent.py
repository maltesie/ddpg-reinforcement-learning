import numpy as np
from collections import defaultdict

import sys
sys.path.append('..') # sorry... :)
from wondevwoman import weight_logger


def sigmoid(x):
    '''an activation function'''
    return 1. / (1. + np.exp(-x))

def d_sigmoid(x):
    '''its derivative'''
    # twice as fast as doing: np.exp(x) / ((np.exp(x) + 1.) ** 2.)
    s = sigmoid(x)
    return s * (1 - s)

def rect(x, leakiness=0.1):
    '''(leaky) rectifier (ReLU) activation function'''
    x[x < 0] *= leakiness
    return x

def d_rect(x, leakiness=0.1):
    '''its derivative'''
    negatives = x < 0
    x[:] = 1
    x[negatives] = leakiness
    return x


class Layer(object):
    def __init__(self, shape, activation_function, activation_function_derivative, rms_decay_rate=0.99, learning_rate=0.000333):
        assert(len(shape) == 2)
        self.shape = shape
        self.neurons = shape[0]
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.rms_decay_rate = rms_decay_rate
        self.learning_rate = learning_rate
        # layer weights
        self.W = np.random.randn(*shape) / np.sqrt(shape[1])
        # gradient buffer (to be aggregated until update_weights())
        self.dW = np.zeros_like(self.W)
        # rms gradient descent optimizer
        self.rmsprop = np.zeros_like(self.W)

    def update_weights(self):
        '''flushes the gradient buffer'''
        # do fancy rmsprop-optimized gradient descent
        self.rmsprop = self.rms_decay_rate * self.rmsprop + (1 - self.rms_decay_rate) * self.dW**2
        self.W += self.learning_rate * self.dW / (np.sqrt(self.rmsprop) + 0.00001)
        # clear gradient buffer
        self.dW = np.zeros_like(self.dW)


class NeuralNet(object):
    def __init__(self, input_dimensions):
        self.input_dimensions = input_dimensions
        self.layers = []

    def add_layer(self, neurons, activation_function, activation_function_derivative):
        prev_layer_neurons = self.input_dimensions if len(self.layers) == 0 else self.layers[-1].neurons
        layer = Layer((neurons, prev_layer_neurons), activation_function, activation_function_derivative)
        self.layers.append(layer)

    def forward(self, x):
        '''forward pass through the net with input x. returns all layer states from input -> output.'''
        states = [x]
        for layer in self.layers:
            state = layer.activation_function(layer.W.dot(states[-1]))
            states.append(state)
        return states

    def backpropagate(self, states, error):
        '''backpropagation. stores weight deltas in the layers' gradient buffer respectively.'''
        for state, layer in reversed(list(zip(states, self.layers))):
            dW = np.outer(error, state)
            layer.dW += dW
            error = error.dot(layer.W) * layer.activation_function_derivative(state)

    def backpropagate_batch(self, states_batch, error_batch):
        for states, error in zip(states_batch, error_batch):
            self.backpropagate(states, error)

    def update_weights(self):
        [l.update_weights() for l in self.layers]


class Agent(object):
    def __init__(self,
            batch_size = 1,
            learning_rate = 0.000333,
            rms_decay_rate = 0.99,
            hidden_neurons1 = 100,
            hidden_neurons2 = 100,
            learn_model = True,
            rect_leakiness=0.1,
            log_filename = None):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rms_decay_rate = rms_decay_rate
        self.learn_model = learn_model
        self.rect_leakiness = rect_leakiness

        # dim of observation vector (input)
        i = 4
        # dim of action probability vector (output)
        if learn_model:
            o = 1 + i # 1 action + model
        else:
            o = 1 # only action

        self.net = NeuralNet(i)
        self.net.add_layer(hidden_neurons1, lambda x: rect(x, rect_leakiness), lambda x: d_rect(x, rect_leakiness))
        self.net.add_layer(hidden_neurons2, lambda x: rect(x, rect_leakiness), lambda x: d_rect(x, rect_leakiness))
        self.net.add_layer(o, sigmoid, d_sigmoid)

        # value storage to be filled during a match
        self.history = defaultdict(list)

        self.log = None
        if log_filename:
            self.log = weight_logger.WeightLogger(log_filename, overwrite=True)

        # number of played/trained games
        self.nb_games = 0

    def get_normalized_state(self, x):
        '''shifts and scales the 0..1 net output vector to match the state bounds'''
        y = np.zeros_like(x)
        y[0] = (x[0] + 2.4) / 4.8
        y[1] = (x[1] + 4.) / 8.
        y[2] = (x[2] + 0.20943951) / 0.41887902
        y[3] = (x[3] + 4.) / 8.
        return y

    def get_discounted_rewards(self):
        '''processes the rewards from history and returns them suited for training'''
        gamma = 0.99 # reward back-through-time aggregation ratio
        r = np.array(self.history['r'], dtype=np.float64)

        if np.all(r == 1):
            return None

        for i in range(len(r) - 2, -1, -1):
            r[i] = r[i] + gamma * r[i+1]

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        r -= np.mean(r)
        r /= np.std(r)

        return r

    def get_action(self, observation, training=True):
        '''asks the net what action to do given an `observation` and does some
           book-keeping except when training is disabled'''
        _, h1, h2, p = self.net.forward(observation)
        action = 0 if p[0] < np.random.random() else 1
        if training:
            self.history['x'].append(observation)
            self.history['h1'].append(h1)
            self.history['h2'].append(h2)
            self.history['p'].append(p)
            self.history['a'].append(action)
        return action

    def add_feedback(self, reward, next_observation):
        '''appends a reward and the actual next observation to history'''
        self.history['r'].append(reward)
        self.history['xn'].append(next_observation)

    def end_game(self):
        '''shuts down a game and triggers a learning step when needed'''
        rewards = self.get_discounted_rewards()
        if rewards is not None:
            # compute errors
            actions = np.array(self.history['a'])
            ps = np.array(self.history['p']).T
            next_observations = np.array(self.history['xn']).T
            if self.learn_model:
                # action gradient and next observation estimation error
                errors = np.vstack(([rewards * (actions - ps[0])], self.get_normalized_state(next_observations) - ps[1:])).T
            else:
                # model-free variant
                errors = rewards * (actions - ps)

            states = zip(self.history['x'], self.history['h1'], self.history['h2'])
            self.net.backpropagate_batch(states, errors)

        self.history.clear()

        if self.nb_games % self.batch_size == 0:
            # time to learn
            self.net.update_weights()
            if self.log:
                self.log.log([l.W for l in self.net.layers])

    def train_games(self, environment, n):
        '''trains on `n` games in `environment`'''
        nb_stepss = [] # list of steps achieved per game
        for i in range(n):
            observation = environment.reset()
            done = False
            nb_steps = 0
            while not done:
                action = self.get_action(observation)

                observation, reward, done, info = environment.step(action)
                nb_steps += 1

                # `reward` is always 1, so use `done` instead
                self.add_feedback(-1 if done and nb_steps < 500 else 1, observation)

            # store
            nb_stepss.append(nb_steps)
            self.nb_games += 1

            self.end_game()

        return nb_stepss

    def evaluate_games(self, environment, n):
        '''performs `n` runs in `environment` and returns the average score'''
        nb_stepss = [] # list of steps achieved per game
        for i in range(n):
            observation = environment.reset()
            done = False
            nb_steps = 0
            while not done:
                action = self.get_action(observation, training=False)
                observation, _, done, _ = environment.step(action)
                nb_steps += 1
            nb_stepss.append(nb_steps)
        return sum(nb_stepss) / float(n)

