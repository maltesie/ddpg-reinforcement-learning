'''
An agent training a neural net with reinforcement learning

http://karpathy.github.io/2016/05/31/rl/
https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
https://github.com/llSourcell/how_to_do_math_for_deep_learning/blob/master/demo.py
http://cs231n.github.io/neural-networks-2/
'''

import sys
import math
import random
import time
from collections import defaultdict
import numpy as np


def sigmoid(x):
    '''an activation function'''
    #with np.errstate(over='raise'): # there are overflows happening in exp due to high values from `net_forward`
    return 1. / (1. + np.exp(-x))

def d_sigmoid(x):
    '''its derivative'''
    return np.exp(x) / ((np.exp(x) + 1.) ** 2.)


def get_direction(src, dst):
    '''returns the geographic / compass direction from position `src` to position `dst`'''
    x1, y1 = src
    x2, y2 = dst
    result = ''
    if y2 < y1: result += 'N'
    if y2 > y1: result += 'S'
    if x2 < x1: result += 'W'
    if x2 > x1: result += 'E'
    return result


class Player:
    '''stores the position of a figure and whether we can control it'''
    def __init__(self, friendly, position=None):
        self.friendly = friendly
        self.position = None
        if position is not None:
            self.set_position(position)

    def set_position(self, position):
        self.position = list(position)


class RLAgent:
    '''main AI unit, storing the world and making decisions'''
    def __init__(self, nb_players_per_side=1):
        # number of games to be played per "learning step"
        self.batch_size = 10
        self.learning_rate = 0.003 # 0.0001
        self.decay_rate = 0.99 # controls rmsprop sum leakiness
        # dim of observation vector (input)
        i = 50 # cell heights (5x5), occupancy (5x5)
        # number of hidden neurons
        h = 200
        # dim of action probability vector (output)
        o = 17 # push or build (1), first direction (8), second direction (8)
        # init weights
        self.W1 = np.random.randn(h, i) / np.sqrt(i)
        self.W2 = np.random.randn(o, h) / np.sqrt(h)
        # gradient buffers for accumulating the batch
        self.dW1 = np.zeros_like(self.W1)
        self.dW2 = np.zeros_like(self.W2)
        # rmsprop buffer (gradient descent optimizer)
        self.rmspropW1 = np.zeros_like(self.W1)
        self.rmspropW2 = np.zeros_like(self.W2)
        # value storage to be filled during a match
        self.history = defaultdict(list)

        self.nb_players_per_side = nb_players_per_side
        self.games_played = 0
        self.reset_game()

    def reset_game(self):
        '''sets the agent up for a new game'''
        self.world = None
        self.score = 0
        self.players = [Player(True) for _ in range(self.nb_players_per_side)] + [Player(False) for _ in range(self.nb_players_per_side)]

    def set_world(self, world):
        '''from the game's world row strings'''
        to_int = lambda c: 4 if c == '.' else int(c)
        self.world = [list(map(to_int, line)) for line in world]

    def set_reward(self, scores):
        '''own score first, opponent's score second'''
        # subtract opponent's score from our score
        score = scores[0] - 0. * scores[1] # TODO: tweak
        self.history['r'].append(score - self.score)
        self.score = score

    def get_cell_player_idx(self, position):
        '''returns the player index if there is a player blocking the cell, -1 otherwise'''
        for i, player in enumerate(self.players):
            if player.position == list(position):
                return i
        return -1

    def get_observation_vector(self, player_idx):
        '''5x5 world block centered around specified player'''
        # TODO: enlarge "visible" block?
        px, py = self.players[player_idx].position
        result = np.empty(5 * 5 * 2)
        # world map
        i = 0
        for x in range(px - 2, px + 3):
            for y in range(py - 2, py + 3):
                if x < 0 or y < 0 or x >= len(self.world) or y >= len(self.world):
                    result[i] = -1 # out of world bounds: blocked cell
                else:
                    l = self.world[y][x]
                    result[i] = l if l < 4 else -1
                i += 1
        # players
        for x in range(px - 2, px + 3):
            for y in range(py - 2, py + 3):
                player_idx = self.get_cell_player_idx((x, y))
                if player_idx == -1:
                    result[i] = 0 # no player
                else:
                    result[i] = 1 if self.players[player_idx].friendly else -1
                i += 1
        return result

    def net_forward(self, x):
        '''forward pass through the net. returns action probability vector and hidden state.'''
        # hidden layer
        h = np.dot(self.W1, x)
        # rectifier function (ReLU)
        h[h < 0] = 0
        # output layer
        p = np.dot(self.W2, h)
        # sigmoid activation function
        p = sigmoid(p)
        return p, h

    def net_backward(self, disc_rs):
        '''does backpropagation with the accumulated value history and processed rewards and returns net gradients'''

        dW1, dW2 = np.zeros_like(self.W1), np.zeros_like(self.W2)

        # iterate over history of hiddens states, delta-probabilities, observations and discounted rewards
        for h, dp, x, disc_r in zip(self.history['h'], self.history['dp'], self.history['x'], disc_rs):
            dpr = dp * disc_r       # k2_delta = k2_error*nonlin(k2,deriv=True)
            dW2 += np.dot(dpr[:, np.newaxis], h[:, np.newaxis].T)     # k1.T.dot(k2_delta)
            dh = dpr.dot(self.W2)   # k1_error = k2_delta.dot(syn1.T)
            dh[h <= 0] = 0          # chain rule: set dh (outer) to 0 where h (inner) is <= 0, k1_delta = k1_error * nonlin(k1,deriv=True)
            dW1 += np.dot(x[:, np.newaxis], dh[:, np.newaxis].T).T  # k0.T.dot(k1_delta)

        return dW1, dW2

        ''' my first try to implemenmt backprop, probably very similar to the one above...
        # hidden state history
        h = np.vstack(self.history['h'])
        #print(h.shape)
        # gradient towards chosen action to be chosen modulated by reward
        dpr = np.vstack(self.history['dp']) * disc_r[:, np.newaxis]     # k2_delta = k2_error*nonlin(k2,deriv=True)
        #print(dpr.shape)

        # backprop
        # TODO: where's derivations? sigmoid?
        #dW2 = np.dot(h.T, dpr) # TODO: had to transpose to make it match W2
        dW2 = dpr.T.dot(h)      # k1.T.dot(k2_delta)
        #print(dW2.shape)

        #dh = np.outer(dpr, self.W2) # TODO: I had to replace `outer` by `dot` to make it work... doesn't this mess it up?
        dh = dpr.dot(self.W2)   # k1_error = k2_delta.dot(syn1.T)
        #print(self.W2.shape)
        dh[h <= 0] = 0

        x = np.vstack(self.history['x'])
        dW1 = np.dot(dh.T, x)
        return dW1, dW2
        '''

    def get_action_from_probability(self, p):
        '''randomly create an action from a given probability vector. returns choice vector and action.'''
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        action_type_p = 0. if p[0] < np.random.rand() else 1.
        action_type = 'MOVE&BUILD' if action_type_p == 0 else 'PUSH&BUILD'
        i = np.random.choice(range(8), p=p[1:9] / np.sum(p[1:9]))
        direction1_p = np.zeros(8)
        direction1_p[i] = 1.
        direction1 = directions[i]
        i = np.random.choice(range(8), p=p[9:17] / np.sum(p[9:17]))
        direction2_p = np.zeros(8)
        direction2_p[i] = 1.
        direction2 = directions[i]
        return np.array(np.hstack(([action_type_p], direction1_p, direction2_p))), (action_type, 0, direction1, direction2)

    def get_action(self):
        '''returns an action based on net and world'''
        x = self.get_observation_vector(0)
        p, h = self.net_forward(x)
        y, action = self.get_action_from_probability(p)
        # store values for backpropagation
        self.history['x'].append(x)
        self.history['h'].append(h)
        self.history['dp'].append(y - p)
        return action

    def discount_rewards(self):
        '''returns the processed (discounted) rewards'''
        r = np.array(self.history['r'], dtype=np.float64)
        gamma = 0.75 # propagate that portion of a reward back in time
        for i in range(len(r) - 2, -1, -1):
            if r[i+1] != -5: # don't spread the punishment for an illegal action back in time
                r[i] += gamma * r[i+1]
        return r

    def end_match(self):
        '''to be called when a game is over'''
        illegal = self.history['r'][-1] == -5

        if len(self.history['r']) > 1 and any(self.history['r']):  # 0-matches can happen
            self.games_played += 1
            # preprocess reward
            disc_r = self.discount_rewards()
            disc_r -= np.mean(disc_r)
            std = np.std(disc_r)
            assert(std != 0)
            disc_r /= std
            # get gradients with backprop and pimped rewards
            dW1, dW2 = self.net_backward(disc_r)
            # aggregate gradients for later use
            self.dW1 += dW1
            self.dW2 += dW2
        self.history.clear()

        if self.games_played % self.batch_size == 0:
            # time to learn
            # do fancy rmsprop-optimized gradient descent
            self.rmspropW1 = self.decay_rate * self.rmspropW1 + (1 - self.decay_rate) * self.dW1**2
            self.W1 += self.learning_rate * self.dW1 / (np.sqrt(self.rmspropW1) + 0.00001)
            # clear gradient buffer
            self.dW1 = np.zeros_like(self.dW1)
            # and second layer
            self.rmspropW2 = self.decay_rate * self.rmspropW2 + (1 - self.decay_rate) * self.dW2**2
            self.W2 += self.learning_rate * self.dW2 / (np.sqrt(self.rmspropW2) + 0.00001)
            self.dW2 = np.zeros_like(self.dW2)

        return illegal

    def shutdown(self):
        '''save state method called when ending a training session'''
        pass

    def resume(self):
        '''restore state method called when resuming a training session'''
        pass
