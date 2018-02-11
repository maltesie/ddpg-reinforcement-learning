'''simple policy gradient reinforcement learning on opengym's cart pole task using tensorflow'''

import tensorflow as tf
import numpy as np
from collections import defaultdict

import train_model

def exp_anneal(gamma, a, b):
    '''returns an exponentially decreasing value between `a` > `b` or
       an exponentially increasing value between `a` < `b` for progress `gamma` [0 .. 1]'''
    a, b = np.log(a), np.log(b)
    return np.exp(gamma * (b - a) + a)


class Agent(object):
    def __init__(self,
            batch_size = 1,
            learning_rate = 0.01,
            rms_decay_rate = 0.99,
            nb_world_features = 32,     # number of first layer's neurons
            rect_leakiness = 0.01,
            learn_model = 'delta',      # ['none', 'delta', 'absolute']
            sample_model = True,
            model_training_noise = 10.0,
            model_training_noise_decay = 0.99,
            model_afunc = None,
            replay_buffer_size = 5000,
            multitask = 'none',         # ['none', 'delta', 'absolute']
            random_seed = None):

        assert(batch_size == 1) # not yet supported

        learn_model = str(learn_model).lower()
        assert(learn_model in ['none', 'delta', 'absolute'])

        multitask = str(multitask).lower()
        assert(multitask in ['none', 'delta', 'absolute'])

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.rms_decay_rate = rms_decay_rate
        self.nb_world_features = nb_world_features
        self.rect_leakiness = rect_leakiness
        self.learn_model = learn_model
        self.sample_model = sample_model
        self.model_training_noise = model_training_noise
        self.model_training_noise_decay = model_training_noise_decay
        self.replay_buffer_size = replay_buffer_size
        self.multitask = multitask

        tf.reset_default_graph()

        if random_seed:
            tf.set_random_seed(random_seed)

        # observation input
        self.net_xs = tf.placeholder(tf.float32, [None, 4], "X")

        # backprop placeholders
        self.net_rewards = tf.placeholder(tf.float32, [None], "R")
        self.net_actions = tf.placeholder(tf.float32, [None], "A")
        self.net_dxs = tf.placeholder(tf.float32, [None, 4], "dX")

        # net 1: action probabilities
        net_world_features = tf.contrib.layers.fully_connected(self.net_xs, nb_world_features, lambda x: tf.nn.leaky_relu(x, alpha=self.rect_leakiness))
        hidden_layer2 = tf.contrib.layers.fully_connected(net_world_features, nb_world_features, lambda x: tf.nn.leaky_relu(x, alpha=self.rect_leakiness))
        self.net_aps = tf.contrib.layers.fully_connected(hidden_layer2, 1, tf.nn.sigmoid)
        # for multitask
        self.net_dxes_multitask = tf.contrib.layers.fully_connected(hidden_layer2, 4, None)

        # net 2: delta-x estimates (model)
        combined_input = tf.concat([self.net_xs, tf.expand_dims(self.net_actions, axis=1)], axis=1)
        hidden_layer1 = tf.contrib.layers.fully_connected(combined_input, nb_world_features * 10, model_afunc)
        # TODO: maybe second hidden layer
        self.net_dxes = tf.contrib.layers.fully_connected(hidden_layer1, 4, None)

        # policy loss functions
        self.loss_actions = -tf.reduce_mean(self.net_rewards * tf.log(tf.multiply(1 - self.net_actions, 1 - self.net_aps[:, 0]) + tf.multiply(self.net_actions, self.net_aps[:, 0])))
        self.loss_actions_dxs_multitask = self.loss_actions + tf.reduce_mean(tf.squared_difference(self.net_dxs, self.net_dxes_multitask))
        # model loss function
        self.loss_dxs = tf.reduce_mean(tf.squared_difference(self.net_dxs, self.net_dxes))

        # training methods
        if multitask == 'none':
            # only minimize policy loss
            self.train_policy = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.rms_decay_rate).minimize(self.loss_actions)
        else:
            # minimize policy and observation prediction loss
            self.train_policy = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.rms_decay_rate).minimize(self.loss_actions_dxs_multitask)
        self.train_model = tf.train.RMSPropOptimizer(learning_rate=.1 * self.learning_rate, decay=self.rms_decay_rate).minimize(self.loss_dxs)

        self.net_session = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # value storage to be filled during a match
        self.history = defaultdict(list)

        # experience storage remembered forever for learning (key -> list of lists of vectors)
        self.experience = defaultdict(list)

        try:
            self.model_test_set = train_model.read_trajectories('cartpole-trajectories.txt', 1)[1]
            self.model_errors = []
        except IOError as e:
            self.model_test_set = None

        # number of played/trained games in the real environment
        self.nb_games = 0
        self.nb_phonies = 0

    def estimate_next_observations(self, xs, actions):
        '''returns the nxes (next x estimates)'''
        dxs = self.net_dxes.eval(feed_dict={self.net_xs: xs, self.net_actions: actions})
        if self.learn_model == 'delta':
            return xs + dxs
        elif self.learn_model == 'absolute':
            return dxs
        else:
            raise RuntimeError('learn_model was set to "%s" while trying to estimate observations' % self.learn_model)

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

        if self.model_test_set:
            plt.xlabel('number of episodes')
            plt.ylabel('model error')
            plt.plot(self.model_errors)
            plt.show()

    def sample_experience(self, n, noise=0.):
        '''returns `n` random experience data points, optionally with gaussian noise of
           standard variances taken from the respective sample dimensions and scaled by `noise`'''
        # TODO: task agnosticity
        xs, actions, dxs = np.empty((n, 4)), np.empty(n), np.empty((n, 4))
        for i in range(n):
            t = np.random.randint(len(self.experience['xs']))
            xs[i] = self.experience['xs'][t]
            actions[i] = self.experience['actions'][t]
            if self.learn_model == 'delta':
                dxs[i] = self.experience['nxs'][t] - xs[i]
            elif self.learn_model == 'absolute':
                dxs[i] = self.experience['nxs'][t]
            else:
                raise RuntimeError('learn_model was set to "%s" while trying to sample experience' % self.learn_model)

        # noise!
        if noise > 0:
            stds = np.std(xs, axis=0)
            for i, std in enumerate(stds):
                xs[:, i] += np.random.normal(0, std * noise, len(xs))

        return xs, actions, dxs

    def get_model_error(self, xs, actions, nxs):
        nxs = np.asarray(nxs)
        nxes = self.estimate_next_observations(xs, actions)
        # normalize
        stds = np.std(nxs, axis=0)
        nxs /= stds
        nxes /= stds
        # find candidates with lowest error
        return np.mean((nxs - nxes) ** 2)

    def forget_experience(self, target_size):
        '''shrinks the replay buffer to `target_size` by removing the items with least error'''
        n = len(self.experience['xs'])
        nb_remove = n - target_size
        if nb_remove <= 0:
            return

        nxs = np.asarray(self.experience['nxs'])

        # compute next x estimates
        nxes = self.estimate_next_observations(self.experience['xs'], self.experience['actions'])
        # normalize
        stds = np.std(nxs, axis=0)
        nxs /= stds
        nxes /= stds
        # find candidates with lowest error
        errors_and_indices = list(zip(np.sum((nxs - nxes) ** 2, axis=1), range(n)))
        sorted_indices = list(zip(*sorted(errors_and_indices)))[1]
        to_be_removed = [False] * n
        for i in sorted_indices[nb_remove:]:
            to_be_removed[i] = True
        # remove them, because they're boring
        self.experience['xs'] = [e for i, e in enumerate(self.experience['xs']) if to_be_removed[i]]
        self.experience['actions'] = [e for i, e in enumerate(self.experience['actions']) if to_be_removed[i]]
        self.experience['nxs'] = [e for i, e in enumerate(self.experience['nxs']) if to_be_removed[i]]

        return nb_remove

    def get_action(self, observation, training=True):
        '''samples action based on an `observation` and does book-keeping when `training`. returns best action when not.'''
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
        nb_games_batch = 0 # number of games trained in this batch until now

        sample_model = False

        while nb_games_batch < n:
            # whether to sample the learned model or the real environment
            #sample_model = self.sample_model and (np.random.random() < exp_anneal(max(self.nb_games / 3000., 0), 0.02, 0.5)) # TODO: tune/params
            # phase-based model sampling
            if self.sample_model and (self.nb_games % 20 == 0) and (self.nb_phonies % 20 == 0):
                sample_model = self.nb_games > self.nb_phonies

            if sample_model:
                # TODO: task agnosticity: initial value ranges/distributions are not given -> learn?
                observation = np.random.random(4) * [.1, .1, .1, .1] - [.05, .05, .05, .05]
            else:
                observation = environment.reset()

            done = False
            nb_steps = 0

            while not done:
                action = self.get_action(observation)

                if sample_model:
                    observation = self.estimate_next_observations([observation], [action])[0]
                    # TODO: task agnosticity: does not apply to all tasks
                    done = (
                        not (np.all(observation < environment.observation_space.high) and np.all(observation > environment.observation_space.low))
                        or nb_steps >= 499
                        )
                else:
                    observation, reward, done, info = environment.step(action)
                nb_steps += 1

                # `reward` is always 1, so use `done` instead
                reward = -1. if done and nb_steps < 500 else 1.
                self.history['rewards'].append(reward)
                self.history['nxs'].append(observation)

            # if (self.nb_games) % 100 == 0 and not sample_model:
            #     self.evaluate_model()

            if sample_model:
                self.nb_phonies += 1
            else:
                self.nb_games += 1
                nb_games_batch += 1
                nb_stepss.append(nb_steps) # store number of achieved steps

            # train policy
            discounted_rewards = self.get_discounted_rewards()
            if discounted_rewards is not None: # don't train on a fully successful episode, because all rewards are 1
                feed_dict = {self.net_xs: self.history['xs'],
                    self.net_rewards: discounted_rewards,
                    self.net_actions: self.history['actions']}
                if self.multitask == 'delta':
                    dxs = np.asarray(self.history['nxs']) - np.asarray(self.history['xs'])
                    feed_dict[self.net_dxs] = dxs
                elif self.multitask == 'absolute':
                    feed_dict[self.net_dxs] = self.history['nxs']
                self.net_session.run(self.train_policy, feed_dict=feed_dict)

            if self.learn_model != 'none' and not sample_model:
                # store observation/action trajectory
                self.experience['xs'] += self.history['xs']
                self.experience['actions'] += self.history['actions']
                self.experience['nxs'] += self.history['nxs']

                # remove experience when the buffer gets too big
                if len(self.experience['xs']) >= 1.5 * self.replay_buffer_size:
                    self.forget_experience(self.replay_buffer_size)

                if self.model_test_set:
                    self.model_errors.append(self.get_model_error(*self.model_test_set))
                # train model on random experience
                for _ in range(20): # TODO: param: number of updates
                    xs, actions, dxs = self.sample_experience(100, self.model_training_noise) # TODO: param: batch size
                    self.net_session.run(self.train_model, feed_dict={
                        self.net_xs: xs,
                        self.net_actions: actions,
                        self.net_dxs: dxs})
                    self.model_training_noise *= self.model_training_noise_decay

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
