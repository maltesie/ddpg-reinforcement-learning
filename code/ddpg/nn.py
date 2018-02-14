import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors.kde import KernelDensity

class NN(object):
    """docstring for GP"""
    def __init__(self, space_dim, done_fktn, predict_change=False, sample_rejection=False):
        self.input_dim = space_dim + 1
        self.output_dim = self.input_dim -1
        self.X = None
        self.Y = None
        self.done = done_fktn
        self.type = 'NN'
        self.predict_change = predict_change
        self.sample_rejection = sample_rejection
        self.nb_samples = 6000
        self.kde =  KernelDensity(bandwidth = 10/(space_dim * np.power(1000, 1/space_dim)))
        
    def add_trajectory(self, observations, actions, rewards):
        if self.X is None:
            self.X = np.hstack((observations[:-1], actions))
            if self.predict_change: self.Y = np.hstack((observations[1:]-observations[:-1], rewards))
            else: self.Y = np.hstack((observations[1:], rewards))
        else:
            new_X = np.hstack((observations[:-1], actions))
            if self.sample_rejection: 
                index = self.reject_index(new_X)
                print(index.sum(), 'samples added')
            else: index = np.ones(len(new_X), dtype=bool)
            self.X = np.vstack((self.X, new_X[index]))
            if self.predict_change: self.Y = np.vstack((self.Y, np.hstack((np.asarray(observations[1:][index])-observations[:-1][index], rewards[index]))))
            else: self.Y = np.vstack((self.Y, np.hstack((np.asarray(observations[1:][[index]]), rewards[index]))))

    def train(self):
        train_size = np.min((self.nb_samples, self.X.shape[0]))
        train_index = np.arange(self.X.shape[0], dtype=int)
        np.random.shuffle(train_index)
        train_index = train_index[:train_size]
        self.scaler = StandardScaler()
        self.scaler.fit(self.X[train_index])
        X_train = self.scaler.transform(self.X)
        self.model = MLPRegressor(hidden_layer_sizes=(200), activation='logistic')
        self.model.fit(X_train[train_index], self.Y[train_index])

    def predict(self, observation, action):
        obs = self.scaler.transform(np.asarray([*observation, action]).reshape(1,-1))
        y_pred = self.model.predict(obs).flatten()
        if self.predict_change: state_pred = observation.flatten() + y_pred[:-1]
        else: state_pred = y_pred[:-1]
        reward_pred = y_pred[-1]
        return state_pred, reward_pred, self.done(state_pred)

    def reject_index(self, data):
        self.kde.fit(self.X)
        mins = np.min(self.X, axis=0)
        maxs = np.max(self.X, axis=0)
        means = (mins + maxs) / 2
        scales = np.abs(maxs-mins)
        test = np.random.rand(1000,len(scales))-0.5
        test *= scales
        test += means
        scores = self.kde.score_samples(test)
        max_populated, min_populated = np.max(scores), np.min(scores)
        mean_populated = (max_populated + min_populated) / 2
        scores = self.kde.score_samples(data)
        cut = 1.1 * max_populated - (1 / (1 + np.exp(self.nb_samples/len(self.X)-len(self.X)/self.nb_samples))) * np.abs(max_populated-min_populated)
        return scores < cut