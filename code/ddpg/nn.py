import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor

class NN(object):
    """docstring for GP"""
    def __init__(self, space_dim, done_fktn, reward_fktn):
        self.input_dim = space_dim + 1
        self.output_dim = self.input_dim -1
        self.X = None
        self.Y = None
        self.done = done_fktn
        self.reward = reward_fktn

    def add_trajectory(self, observations, actions):
        if self.X is None:
            self.X = np.hstack((observations, actions))[:-1]
            self.Y = np.asarray(observations[1:])
        else:
            self.X = np.vstack((self.X, np.hstack((observations, actions))[:-1]))
            self.Y = np.vstack((self.Y, observations[1:]))

    def train(self, nb_samples):
        train_size = np.min((nb_samples, self.X.shape[0]))
        train_index = np.arange(self.X.shape[0], dtype=int)
        np.random.shuffle(train_index)
        train_index = train_index[:train_size]
        self.scaler = StandardScaler()
        self.scaler.fit(self.X[train_index])
        X_train = self.scaler.transform(self.X)
        self.model = MLPRegressor(hidden_layer_sizes=(16,16))
        self.model.fit(X_train[train_index], self.Y[train_index])

    def predict(self, observation, action):
        obs = self.scaler.transform(np.asarray([*observation, action]).reshape(1,-1))
        y_pred = self.model.predict(obs)
        return y_pred.flatten(), self.reward(y_pred, action), self.done(y_pred)

    def rollout(self, observation, policy, n):
        trajectory_pred = np.empty(n, self.output_dim)
        current_state = observation
        current_action = policy(current_state)
        for i in range(n):
            trajectory_pred[i] = self.predict(current_state, current_action)
            current_state = trajectory_pred[i]
            current_action = policy(current_state)
        return trajectory_pred