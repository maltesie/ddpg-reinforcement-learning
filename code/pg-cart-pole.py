import gym
import numpy as np
from collections import defaultdict, deque
import time, sys

from wondevwoman import weight_logger, weight_plotter


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
    def __init__(self, nb_neurons, activation_function, activation_function_derivative, rms_decay_rate=0.99, learning_rate=0.000333):
        self.nb_neurons = nb_neurons
        self.shape = [nb_neurons, 0]
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.rms_decay_rate = rms_decay_rate
        self.learning_rate = learning_rate

    def init_weights(self):
        '''creates the arrays neccessary for layer weight management'''
        # `shape` depends on the number of neurons and the number of values received from previous layers
        self.W = np.random.randn(*self.shape) / np.sqrt(self.shape[1])
        # gradient buffer (to be aggregated until update_weights())
        self.dW = np.zeros_like(self.W)
        # rms gradient descent optimizer
        self.rmsprop = np.zeros_like(self.W)

    def update_weights(self):
        '''flushes the gradient buffer'''
        # do fancy rmsprop-optimized gradient descent
        print(self.learning_rate)
        self.rmsprop = self.rms_decay_rate * self.rmsprop + (1 - self.rms_decay_rate) * self.dW**2
        self.W += self.learning_rate * self.dW / (np.sqrt(self.rmsprop) + 0.00001)
        # clear gradient buffer
        self.dW = np.zeros_like(self.dW)


class NeuralNet(object):
    def __init__(self, input_dimensions):
        '''input_dimensions: list of input dimensions'''
        try:
            iter(input_dimensions)
            # multiple inputs
            self.input_dimensions = input_dimensions
            self.nb_inputs = len(input_dimensions)
        except TypeError:
            # only one input
            self.input_dimensions = (input_dimensions,)
            self.nb_inputs = 1
        self.output_layers = []
        self.prevs = defaultdict(list)
        self.nexts = defaultdict(list)
        self.layers = []

    def create_layer(self, nb_neurons, activation_function, activation_function_derivative, **kwargs):
        '''adds a new layer to the net (without connecting it to any others)'''
        layer = Layer(nb_neurons, activation_function, activation_function_derivative, **kwargs)
        self.layers.append(layer)
        return layer

    def share_layer(self, layer):
        '''adds an existing layer object to the net'''
        self.layers.append(layer)

    def connect(self, layer1, layer2, shared=False):
        '''layers can be hidden/output Layer objects or an input index. a shared connection is one which does not change the weight shape.'''
        self.nexts[layer1].append(layer2)
        self.prevs[layer2].append(layer1)
        if not shared:
            # update shape of layer weights
            layer2.shape[1] += layer1.nb_neurons if isinstance(layer1, Layer) else self.input_dimensions[layer1]
        # update output layer list
        if layer1 in self.output_layers:
            self.output_layers.remove(layer1)
        if layer2 not in self.output_layers and len(self.nexts[layer2]) == 0:
            self.output_layers.append(layer2)

    def init_weights(self):
        '''create weight arrays in each layer'''
        for layer in self.layers:
            layer.init_weights()

    def get_prevs_state(self, states, layer):
        '''gathers the states of all previous layers in one vector'''
        prev_states = []
        # allow up to one prev layer with two next layers
        multi_nexts_layer = None
        for prev in self.prevs[layer]:
            assert(len(self.nexts[prev]) > 0)
            if len(self.nexts[prev]) > 1:
                assert(multi_nexts_layer is None)
                multi_nexts_layer = prev
                assert(False) # TODO: does not work yet (second half not handled yet)
            else:
                prev_states.append(states[prev])
        # the layer with multiple next layers (if available) has to be the last one
        if multi_nexts_layer is not None:
            prev_states.append(multi_nexts_layer)

        # concatenate state and cut off the excess (if having a layer with multiple next states)
        s = np.concatenate(prev_states)[:layer.shape[1]]

        return s

    def forward(self, inputs):
        '''forward pass through the net with `inputs`. returns all layer states from inputs -> outputs as dict.'''
        states = {}
        # set input states
        if self.nb_inputs == 1:
            states[0] = inputs
        else:
            for idx, x in enumerate(inputs):
                states[idx] = x
        # add all next-to-input layers to the queue
        q = deque([layer for input_idx in range(self.nb_inputs) for layer in self.nexts[input_idx]])
        while len(q) > 0:
            layer = q.popleft()
            if any((prev not in states) for prev in self.prevs[layer]):
                # previous states not yet computed: postpone
                q.append(layer)
            else:
                # gather states of all previous layers
                s = self.get_prevs_state(states, layer)

                # actual layer step
                states[layer] = layer.activation_function(layer.W.dot(s))

                for next_layer in self.nexts[layer]:
                    q.append(next_layer)

        return states

    def backpropagate(self, states, errors):
        '''backpropagation. gets dict of layer states and list of output layer error vectors.
           stores weight deltas in the layers' gradient buffer respectively.'''
        # turn errors into a dict layer -> error
        errors = {lay: err for lay, err in zip(self.output_layers, errors)}
        q = deque(self.output_layers)
        while len(q) > 0:
            layer = q.popleft()
            # compute weight deltas as product of error and previous layers' state
            s = self.get_prevs_state(states, layer)
            dW = np.outer(errors[layer], s)
            layer.dW += dW
            # compute previous layers' error as product of current error, weights and derivative of state
            prev_error = errors[layer].dot(layer.W) * layer.activation_function_derivative(s)
            # split error vector into previous layers
            i = 0
            for prev_layer in self.prevs[layer]:
                assert(prev_layer not in errors)
                nb_neurons = prev_layer.nb_neurons if isinstance(prev_layer, Layer) else self.input_dimensions[prev_layer]
                errors[prev_layer] = prev_error[i : i + nb_neurons]
                i += nb_neurons
                # add previous layer to queue (except input layers)
                if isinstance(prev_layer, Layer):
                    q.append(prev_layer)

    def backpropagate_batch(self, states_batch, error_batch):
        '''calls backpropagation for each state-error pair in the lists'''
        for states, error in zip(states_batch, error_batch):
            self.backpropagate(states, error)

    def update_weights(self):
        '''flushes the gradient buffers of all layers'''
        for layer in self.layers:
            layer.update_weights()


class Agent(object):
    def __init__(self,
            batch_size = 1,
            learning_rate = 0.000333,
            rms_decay_rate = 0.99,
            hidden_neurons1 = 100,
            hidden_neurons2 = 100,
            hidden_neurons3 = 100,
            learn_model = True,
            rect_leakiness=0.1,
            log_filename = 'pg-cart-pole.log'):

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rms_decay_rate = rms_decay_rate
        self.learn_model = learn_model
        self.rect_leakiness = rect_leakiness

        # dim of observation vector (input)
        i = 4
        # # dim of action probability vector (output)
        # if learn_model:
        #     o = 1 + i # 1 action + model
        # else:
        #     o = 1 # only action

        lrect = lambda x: rect(x, rect_leakiness)
        d_lrect = lambda x: d_rect(x, rect_leakiness)

        self.net_action = NeuralNet(i)      # input: observation (vector of length i)
        self.net_model = NeuralNet((i, 1))  # input: observation + action
        l_observation = self.net_action.create_layer(hidden_neurons1, lrect, d_lrect)
        #self.net_model.share_layer(l_observation)
        #l_observation2 = self.net_model.create_layer(8, lrect, d_lrect, learning_rate=0.005)
        l_to_action = self.net_action.create_layer(hidden_neurons2, lrect, d_lrect)
        #l_to_observation_prediction = self.net_model.create_layer(8, lrect, d_lrect, learning_rate=0.005)
        self.l_p_action = self.net_action.create_layer(1, sigmoid, d_sigmoid)
        self.l_observation_prediction = self.net_model.create_layer(i, lambda x: x, lambda x: 1, learning_rate=0.1)
        #self.l_observation_prediction = self.net_model.create_layer(i, sigmoid, d_sigmoid, learning_rate=0.005)

        self.net_action.connect(0, l_observation)
        self.net_action.connect(l_observation, l_to_action)
        self.net_action.connect(l_to_action, self.l_p_action)

        #self.net_model.connect(0, l_observation, shared=True)
        #self.net_model.connect(l_observation, l_to_observation_prediction)
        # self.net_model.connect(0, l_observation2)
        # self.net_model.connect(l_observation2, l_to_observation_prediction)
        # self.net_model.connect(1, l_to_observation_prediction)
        # self.net_model.connect(l_to_observation_prediction, self.l_observation_prediction)
        self.net_model.connect(0, self.l_observation_prediction)

        self.net_action.init_weights()
        self.net_model.init_weights()

        #h1 = self.net.create_layer(hidden_neurons1, lambda x: rect(x, rect_leakiness), lambda x: d_rect(x, rect_leakiness))
        #h2 = self.net.create_layer(hidden_neurons2, lambda x: rect(x, rect_leakiness), lambda x: d_rect(x, rect_leakiness))
        #self.p_layer = self.net.create_layer(o, sigmoid, d_sigmoid)
        #self.net.connect(0, h1)
        #self.net.connect(h1, h2)
        #self.net.connect(h2, self.p_layer)
        #self.net.init_weights()

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
        states = self.net_action.forward(observation)
        p = states[self.l_p_action]
        action = 0 if p[0] < np.random.random() else 1
        if training:
            self.history['states_action'].append(states)
            self.history['a'].append(action)
            # also predict the next observation
            states = self.net_model.forward((observation, np.asarray([action])))
            self.history['states_model'].append(states)

        return action

    def add_feedback(self, reward, next_observation):
        '''appends a reward and the actual next observation to history'''
        self.history['r'].append(reward)
        self.history['xn'].append(next_observation)

    def end_game(self):
        '''shuts down a game and triggers a learning step when needed'''
        rewards = self.get_discounted_rewards()
        if rewards is not None:
            # compute action error which enforces the action which was taken to be taken
            actions = np.array(self.history['a'])
            action_ps = np.array([states[self.l_p_action] for states in self.history['states_action']]).T
            errors = (rewards * (actions - action_ps)).T
            errors = errors[:, np.newaxis, :]           # one error vector per output layer
            # train policy
            self.net_action.backpropagate_batch(self.history['states_action'], errors)

        # compute observation estimation error
        observations = np.array([states[0] for states in self.history['states_model']])
        observation_predictions = np.array([states[self.l_observation_prediction] for states in self.history['states_model']])
        errors = (observations - observation_predictions)
        avg_error = errors #self.get_normalized_state(errors.T)
        avg_error = np.mean(avg_error)
        print(abs(avg_error), observations[0], observation_predictions[0], errors[0])
        errors = errors[:, np.newaxis, :]
        # train model
        self.net_model.backpropagate_batch(self.history['states_model'], errors)

        self.history.clear()

        if self.nb_games % self.batch_size == 0:
            # time to learn
            self.net_action.update_weights()
            self.net_model.update_weights() # TODO didnt update the WHOLE time
            if self.log:
                self.log.log([l.W for l in self.net_model.layers])

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


if len(sys.argv) > 1:
    mode = sys.argv[1]
    if not mode in ['demo', 'evaluate']:
        print('usage: ' + sys.argv[0] + ' [demo|evaluate]')
        sys.exit(1)
else:
    mode = 'evaluate' # 'demo', 'evaluate'

env = gym.make('CartPole-v1')

# hyperparameters
param = {'learn_model': False, 'rect_leakiness': 0.1, 'log_filename': 'model.log'}


if mode == 'demo':
    agent = Agent(**param)
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
    nb_trainings = 100
    # number of games to train on
    nb_games = 2000
    # steps size (in games) at which to perform an evaluation
    batch_size = 100
    # instances to evaluate and average
    nb_evaluations = 10

    scoress = [[] for _ in range(nb_trainings)]
    for t in range(nb_trainings):
        print('\rtrainging instance %i/%i' % (t + 1, nb_trainings))
        agent = Agent(**param, log_filename=False)
        while agent.nb_games < nb_games:
            print('\rgame %i/%i' % (agent.nb_games, nb_games), end='', flush=True)
            agent.train_games(env, batch_size)
            avg_steps_achieved = agent.evaluate_games(env, nb_evaluations)
            scoress[t].append(avg_steps_achieved)
    # include hyperparameters in filename where to store the individual scores of each training run
    filename = 'eval-scores-%s.txt' % '.'.join([str(k) + '-' + str(v) for k, v in param.items()])
    with open(filename, 'w') as f:
        [f.write(str(scores) + '\n') for scores in scoress]
    # compute mean of scores across training runs and standard error of the mean
    scoress = np.array(scoress, dtype=np.float64)
    scores = np.mean(scoress, axis=0)
    stderr = scoress.std(axis=0, ddof=1) / np.sqrt(scoress.shape[0])
    print()
    print(param)
    print('scores:', list(scores))
    print('stderr:', list(stderr))
    X = range(batch_size, nb_games + batch_size, batch_size)
    plt.plot(X, scores)
    plt.errorbar(X, scores, stderr)
    plt.show()
