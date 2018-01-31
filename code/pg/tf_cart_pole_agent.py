'''simple policy gradient reinforcement learning on opengym's cart pole task using tensorflow'''

import tensorflow as tf
import numpy as np
from collections import defaultdict

class Agent(object):
    def __init__(self,
            batch_size = 1,
            learning_rate = 0.01,
            rms_decay_rate = 0.99,
            nb_world_features = 8,  # number of first layer's neurons
            learn_model=True,
            rect_leakiness=0.1):

        assert(batch_size == 1) # not yet supported

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rms_decay_rate = rms_decay_rate
        self.nb_world_features = nb_world_features
        self.learn_model = learn_model
        self.rect_leakiness = rect_leakiness

        # neural net setup:
        # x -> W1 -> W2 -> ap
        #       |
        #       v
        # a -> W3 -> dx

        tf.reset_default_graph()

        # observation input and trainable weights
        self.net_xs = tf.placeholder(tf.float32, [None, 4], "X")
        self.net_W1 = tf.get_variable("W1", shape=[4 + 1, nb_world_features], initializer=tf.contrib.layers.xavier_initializer())
        self.net_W2_1 = tf.get_variable("W2_1", shape=[nb_world_features + 1, nb_world_features + 1], initializer=tf.contrib.layers.xavier_initializer())
        self.net_W2_2 = tf.get_variable("W2_2", shape=[nb_world_features + 2, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.net_W3_1 = tf.get_variable("W3_1", shape=[nb_world_features + 2, nb_world_features + 2], initializer=tf.contrib.layers.xavier_initializer())
        self.net_W3_2 = tf.get_variable("W3_2", shape=[nb_world_features + 3, 4], initializer=tf.contrib.layers.xavier_initializer())

        # backprop placeholders
        self.net_rewards = tf.placeholder(tf.float32, [None], "R")
        self.net_actions = tf.placeholder(tf.float32, [None], "A")
        self.net_dxs = tf.placeholder(tf.float32, [None, 4], "dX")

        # shared first hidden layer: "world features"
        self.net_world_features = tf.nn.leaky_relu(tf.matmul(tf.pad(self.net_xs, [[0, 0], [0, 1]], constant_values=1), self.net_W1), alpha=self.rect_leakiness)

        # output: action probabilities
        self.net_aps = tf.nn.sigmoid(   # third layer, sigmoid is good for classification
            tf.matmul(
                tf.pad(                 # bias
                    tf.nn.leaky_relu(   # second layer
                        tf.matmul(
                            tf.pad(self.net_world_features, [[0, 0], [0, 1]], constant_values=1),   # bias
                            self.net_W2_1
                        ),
                        alpha=self.rect_leakiness
                    ), [[0, 0], [0, 1]], constant_values=1
                ),
                self.net_W2_2
            )
        )

        # output: next x estimates
        self.net_dxes = tf.matmul(      # third layer, no activation function
            tf.pad(                     # bias
                tf.nn.leaky_relu(       # second layer, relu seems better for general regression
                    tf.matmul(
                        tf.pad(         # bias
                            tf.concat([self.net_world_features, tf.expand_dims(self.net_actions, axis=1)], axis=1), # merge first layer with action
                            [[0, 0], [0, 1]], constant_values=1
                        ),
                        self.net_W3_1
                    ),
                    alpha=self.rect_leakiness
                ),
                [[0, 0], [0, 1]], constant_values=1
            ),
            self.net_W3_2
        )

        # loss functions of the outputs above
        self.loss_actions = -tf.reduce_mean(self.net_rewards * tf.log(tf.multiply(1 - self.net_actions, 1 - self.net_aps[:, 0]) + tf.multiply(self.net_actions, self.net_aps[:, 0])))
        self.loss_dxs = tf.reduce_mean(tf.squared_difference(self.net_dxs, self.net_dxes))

        # training methods
        self.train_policy = tf.train.RMSPropOptimizer(learning_rate=.01, decay=.99).minimize(self.loss_actions)
        self.train_model = tf.train.RMSPropOptimizer(learning_rate=.001, decay=.99).minimize(self.loss_dxs)

        self.net_session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # value storage to be filled during a match
        self.history = defaultdict(list)

        # experience storage remembered forever for learning (key -> list of lists of vectors)
        self.experience = defaultdict(list)

        # number of played/trained games
        self.nb_games = 0

    def estimate_next_observations(self, xs, actions):
        '''returns the nxes (next x estimates)'''
        dxs = self.net_dxes.eval(feed_dict={self.net_xs: xs, self.net_actions: actions})
        return xs + dxs

    def evaluate_model(self):
        '''plots the observation history, the prediction of next observations and a chain prediction only depending on the first observation'''
        import matplotlib.pyplot as plt

        # real xs
        xs = np.asarray([self.history['xs'][0]] + self.history['nxs'])
        # bulk prediction
        nxes_bulk = self.estimate_next_observations(self.history['xs'], self.history['actions'])
        # chain prediction
        nxes_chain = []
        x = self.history['xs'][0]
        for action in self.history['actions']:
            x = self.estimate_next_observations([x], [action])[0]
            nxes_chain.append(x)
        nxes_chain = np.asarray(nxes_chain)

        # plot!
        plt.plot(range(len(xs)), xs[:, 0], color='black', label='ground truth')
        # create axis for predictions starting at 1, not 0
        X = range(1, nxes_bulk.shape[0] + 1)
        plt.plot(X, nxes_bulk[:, 0], color='#0055AA', label='bulk prediction')
        plt.plot(X, nxes_chain[:, 0], color='#CC0022', label='chain prediction')
        plt.legend()
        plt.show()

    def sample_experience(self, n):
        '''returns `n` random experience data points'''
        result = []
        for _ in range(n):
            i = np.random.randint(len(self.experience['xs']))
            t = np.random.randint(len(self.experience['xs'][i]) - 1)
            x = self.experience['xs'][i][t]
            action = self.experience['actions'][i][t]
            dx = self.experience['xs'][i][t + 1] - x
            result.append([x, action, dx])
        return result

    def get_action(self, observation, training=True):
        '''when `training`, samples action based on an `observation` and does book-keeping. returns best action when not.'''
        # get action probability
        y = self.net_aps.eval(feed_dict={self.net_xs: [observation]})[0][0]
        if training:
            # sample action
            action = 0 if y < np.random.random() else 1
            # book keeping
            self.history['xs'].append(observation)
            self.history['actions'].append(action)
        else:
            # do best action
            action = 0 if y < 0.5 else 1
        return action

    def get_discounted_rewards(self):
        '''adds up each reward and its following rewards with a discount factor. centers and normalizes.'''
        gamma = 0.99 # reward back-through-time aggregation ratio
        r = np.asarray(self.history['rewards'], dtype=np.float64)

        if np.all(r == 1):
            return None

        for i in range(len(r) - 2, -1, -1):
            r[i] = r[i] + gamma * r[i+1]

        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        r -= np.mean(r)
        r /= np.std(r)

        return r

    def train_games(self, environment, n):
        '''trains on `n` games in `environment`'''
        nb_stepss = [] # list of steps achieved per episode
        for _ in range(n):
            observation = environment.reset()
            done = False
            nb_steps = 0
            while not done:
                action = self.get_action(observation)

                observation, reward, done, info = environment.step(action)
                nb_steps += 1

                # `reward` is always 1, so use `done` instead
                reward = -1. if done and nb_steps < 500 else 1.
                self.history['rewards'].append(reward)
                self.history['nxs'].append(observation)

            # store number of achieved steps
            nb_stepss.append(nb_steps)

            # if self.nb_games % 100 == 0:
            #     self.evaluate_model()

            # train policy
            discounted_rewards = self.get_discounted_rewards()
            if discounted_rewards is not None: # don't train on a fully successful episode, because all rewards are 1
                self.net_session.run(self.train_policy, feed_dict={
                    self.net_xs: self.history['xs'],
                    self.net_rewards: discounted_rewards,
                    self.net_actions: self.history['actions']})

            if self.learn_model:
                # store observation/action trajectory
                self.experience['xs'].append(self.history['xs'])
                self.experience['actions'].append(self.history['actions'])

                # train model on random experience
                for _ in range(20): # TODO: param: number of updates
                    xs, actions, dxs = zip(*np.asarray(self.sample_experience(100))) # TODO: param: batch size
                    self.net_session.run(self.train_model, feed_dict={
                        self.net_xs: xs,
                        self.net_actions: actions,
                        self.net_dxs: dxs})

            self.nb_games += 1

            # reset history
            self.history.clear()

        return nb_stepss

    def evaluate_games(self, environment, n):
        '''performs `n` runs in `environment` and returns the average score'''
        nb_stepss = [] # list of steps achieved per game
        for _ in range(n):
            observation = environment.reset() # each new environment is initialized randomly
            done = False
            nb_steps = 0
            while not done:
                action = self.get_action(observation, training=False)
                observation, _, done, _ = environment.step(action)
                nb_steps += 1
            nb_stepss.append(nb_steps)
        return sum(nb_stepss) / float(n)
