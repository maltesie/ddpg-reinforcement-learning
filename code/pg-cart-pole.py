import gym
import numpy as np
from collections import defaultdict
import time

from wondevwoman import weight_logger, weight_plotter


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


class Agent(object):
    def __init__(self,
            batch_size = 1,
            learning_rate = 0.000333,
            rms_decay_rate = 0.99,
            hidden_neurons1 = 100,
            hidden_neurons2 = 100,
            learn_model = True,
            log_filename = 'pg-cart-pole.log'):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rms_decay_rate = rms_decay_rate
        self.learn_model = learn_model

        # dim of observation vector (input)
        i = 4
        # dim of action probability vector (output)
        if learn_model:
            o = 1 + i # 1 action + model
        else:
            o = 1 # only action
        # init weights
        self.W1 = np.random.randn(hidden_neurons1, i) / np.sqrt(i)
        self.W2 = np.random.randn(hidden_neurons2, hidden_neurons1) / np.sqrt(hidden_neurons1)
        self.W3 = np.random.randn(o, hidden_neurons2) / np.sqrt(hidden_neurons2)

        # gradient buffers for accumulating the batch
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)
        self.dW3 = np.zeros_like(self.W3)
        # rmsprop buffer (gradient descent optimizer)
        self.rmspropW1 = np.zeros_like(self.W1)
        self.rmspropW2 = np.zeros_like(self.W2)
        self.rmspropW3 = np.zeros_like(self.W3)
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
        return r

    def net_forward(self, x):
        '''forward pass through the net. returns action probability vector and hidden states'''
        # hidden layer 1
        h1 = np.dot(self.W1, x)
        h1 = rect(h1)
        # hidden layer 2
        h2 = np.dot(self.W2, h1)
        h2 = rect(h2)
        # output layer
        p = np.asarray(np.dot(self.W3, h2)) # enforce array form even if it is a scalar in model-free mode
        p = sigmoid(p)
        return p, h1, h2

    def net_backward(self, rewards):
        '''does backpropagation with the accumulated value history and processed rewards and returns net gradients'''
        dW1, dW2, dW3 = np.zeros_like(self.W1), np.zeros_like(self.W2), np.zeros_like(self.W3)

        # iterate over history of observations, hiddens states, net outputs, actions taken, rewards and next observations
        for x, h1, h2, p, a, r, xn in zip(self.history['x'], self.history['h1'], self.history['h2'], self.history['p'], self.history['a'], rewards, self.history['xn']):
            if self.learn_model:
                # action gradient and next observation estimation error
                dp = np.hstack(([r * (a - p[0])], self.get_normalized_state(xn) - p[1:])) * d_sigmoid(p)
            else:
                # model-free variant
                dp = r * (a - p[0]) * d_sigmoid(p)
            dW3 += np.dot(dp[:, np.newaxis], h2[:, np.newaxis].T)

            dh2 = dp.dot(self.W3)
            dh2 *= d_rect(h2)
            dW2 += np.dot(dh2[:, np.newaxis], h1[:, np.newaxis].T)

            dh1 = dh2.dot(self.W2)
            dh1 *= d_rect(h1) # chain rule: set dh (outer) to 0 where h (inner) is <= 0
            dW1 += np.dot(x[:, np.newaxis], dh1[:, np.newaxis].T).T

        return dW1, dW2, dW3

    def get_action(self, observation, training=True):
        '''asks the net what action to do given an `observation` and does some
           book-keeping except when training is disabled'''
        p, h1, h2 = self.net_forward(observation)
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
        r = self.get_discounted_rewards()
        if r is not None:
            r -= np.mean(r)
            std = np.std(r)
            r /= std
            # get gradients with backprop and pimped rewards
            dW1, dW2, dW3 = self.net_backward(r)
            # aggregate gradients for later use
            self.dW1 += dW1
            self.dW2 += dW2
            self.dW3 += dW3
        self.history.clear()

        if self.nb_games % self.batch_size == 0:
            # time to learn
            self.update_weights()

    def update_weights(self):
        '''flushes the gradient buffers'''
        # do fancy rmsprop-optimized gradient descent
        self.rmspropW1 = self.rms_decay_rate * self.rmspropW1 + (1 - self.rms_decay_rate) * self.dW1**2
        self.W1 += self.learning_rate * self.dW1 / (np.sqrt(self.rmspropW1) + 0.00001)
        # clear gradient buffer
        self.dW1 = np.zeros_like(self.dW1)

        # and second layer
        self.rmspropW2 = self.rms_decay_rate * self.rmspropW2 + (1 - self.rms_decay_rate) * self.dW2**2
        self.W2 += self.learning_rate * self.dW2 / (np.sqrt(self.rmspropW2) + 0.00001)
        self.dW2 = np.zeros_like(self.dW2)

        # third
        self.rmspropW3 = self.rms_decay_rate * self.rmspropW3 + (1 - self.rms_decay_rate) * self.dW3**2
        self.W3 += self.learning_rate * self.dW3 / (np.sqrt(self.rmspropW3) + 0.00001)
        self.dW3 = np.zeros_like(self.dW3)

        if self.log:
            self.log.log((self.W1, self.W2, self.W3))

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


env = gym.make('CartPole-v1')

mode = 'evaluate' # 'demo'

if mode == 'demo':
    agent = Agent()
    try:
        # not the actual batch training size, only for statistics and status updates
        batch_size = 100
        while True:
            nb_steps = agent.train_games(env, batch_size)
            avg_nb_steps = sum(nb_steps) / float(batch_size)
            max_steps = max(nb_steps)
            print('\rgame #%i avg number of steps: %.2f (max: %i)' % (agent.nb_games, avg_nb_steps, max_steps), end='', flush=True)

    except KeyboardInterrupt:
        print('\nTrained %i games. Showtime!' % agent.nb_games)

        observation = env.reset()
        done = False
        while not done:
            env.render()

            action = agent.get_action(observation, training=False)
            observation, reward, done, info = env.step(action)
            time.sleep(0.075)

elif mode == 'evaluate':
    import matplotlib.pyplot as plt
    # instances to train and average
    nb_trainings = 50
    # number of games to train on
    nb_games = 2000
    # steps size (in games) at which to perform an evaluation
    batch_size = 100
    # instances to evaluate and average
    nb_evaluations = 10

    scoress = [[] for _ in range(nb_trainings)]
    for t in range(nb_trainings):
        print('\rtrainging instance %i/%i' % (t, nb_trainings))
        agent = Agent(learn_model=True, log_filename=False)
        while agent.nb_games < nb_games:
            print('\rgame %i/%i' % (agent.nb_games, nb_games), end='', flush=True)
            agent.train_games(env, batch_size)
            avg_steps_achieved = agent.evaluate_games(env, nb_evaluations)
            scoress[t].append(avg_steps_achieved)
    scoress = np.array(scoress, dtype=np.float64)
    scores = np.mean(scoress, axis=0)
    print()
    print(list(scores))
    plt.plot(scores)
    plt.show()
