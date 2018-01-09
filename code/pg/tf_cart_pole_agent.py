'''simple policy gradient reinforcement learning on opengym's cart pole task using tensorflow'''

import tensorflow as tf
import numpy as np
import gym
import time
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
        # a -> W3 -> nx

        tf.reset_default_graph()

        # observation input and trainable weights
        self.net_xs = tf.placeholder(tf.float32, [None, 4])
        self.net_W1 = tf.get_variable("W1", shape=[4, nb_world_features], initializer=tf.contrib.layers.xavier_initializer())
        self.net_W2 = tf.get_variable("W2", shape=[nb_world_features, 1], initializer=tf.contrib.layers.xavier_initializer())
        self.net_W3 = tf.get_variable("W3", shape=[9, 4], initializer=tf.contrib.layers.xavier_initializer())

        # backprop placeholders
        self.net_rewards = tf.placeholder(tf.float32, [None])
        self.net_actions = tf.placeholder(tf.float32, [None])
        self.net_nxs = tf.placeholder(tf.float32, [None, 4])

        # intermediate layer "world features"
        self.net_world_features = tf.nn.leaky_relu(tf.matmul(self.net_xs, self.net_W1), alpha=0.1)

        # output: action probabilities
        self.net_aps = tf.nn.sigmoid(tf.matmul(self.net_world_features, self.net_W2))

        # output: next x estimates
        self.net_nxes = tf.matmul(tf.concat([self.net_world_features, tf.expand_dims(self.net_actions, axis=1)], axis=1), self.net_W3)

        # loss functions of the outputs above
        loss_actions = -tf.reduce_mean(self.net_rewards * tf.log(tf.multiply(1 - self.net_actions, 1 - self.net_aps[:, 0]) + tf.multiply(self.net_actions, self.net_aps[:, 0])))
        loss_nxs = tf.reduce_mean((self.net_nxs - self.net_nxes) ** 2)

        # training methods
        self.train_policy = tf.train.RMSPropOptimizer(learning_rate=.01, decay=.99).minimize(loss_actions)
        self.train_model = tf.train.RMSPropOptimizer(learning_rate=.01, decay=.99).minimize(loss_nxs)

        self.net_session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # value storage to be filled during a match
        self.history = defaultdict(list)

        # number of played/trained games
        self.nb_games = 0

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
                reward = -1. if done else 1.
                self.history['rewards'].append(reward)
                self.history['nxs'].append(observation)

            # store number of achieved steps
            nb_stepss.append(nb_steps)

            # train policy
            discounted_rewards = self.get_discounted_rewards()
            self.net_session.run(self.train_policy, feed_dict={
                self.net_xs: self.history['xs'],
                self.net_rewards: discounted_rewards,
                self.net_actions: self.history['actions']})

            if self.learn_model:
                # train model
                self.net_session.run(self.train_model, feed_dict={
                    self.net_xs: self.history['xs'],
                    self.net_nxs: self.history['nxs'],
                    self.net_actions: self.history['actions']})

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

if __name__ == '__main__':

    env = gym.make('CartPole-v1')

    agent = Agent()
    agent.train_games(env, 500)
    agent.evaluate_games(env, 10)
    exit(0)

    running_mean = 22. # init with rougly the average step count of a random agent

    for i in range(10000): # terminate eventually
        # episode log
        steps = 0
        actions = []
        rewards = []
        xs = []
        nxs = []

        # let's go
        x = env.reset()
        done = False
        while not done:
            #env.render()
            #time.sleep(0.075)

            xs.append(x)
            y = net_aps.eval(feed_dict={net_xs: [x]})[0][0] # get action
            action = 0 if np.random.random() > y else 1
            actions.append(action)

            x, reward, done, info = env.step(action)
            rewards.append(-1. if done else 1.)
            nxs.append(x)
            steps += 1

        running_mean = .99 * running_mean + .01 * steps
        print('\rcurrent step count estimate: %.2f' % running_mean, end='', flush=True)

        # train policy
        discounted_rewards = get_discounted_rewards(rewards)
        net_session.run(train_policy, feed_dict={net_xs: xs, net_rewards: discounted_rewards, net_actions: actions})

        # train model
        net_session.run(train_model, feed_dict={net_xs: xs, net_nxs: nxs, net_actions: actions})
