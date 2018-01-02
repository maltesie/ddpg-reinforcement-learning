import numpy as np

from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor

class NN(object):
	"""docstring for GP"""
	def __init__(self, space_dim):
		self.input_dim = space_dim + 1
		self.output_dim = self.input_dim -1
		self.X = None
		self.Y = None

	def add_trajectory(self, observations, actions):
		if self.X is None:
			self.X = np.hstack((observations, actions))[:-1]
			self.Y = np.asarray(observations[1:])
		else:
			self.X = np.vstack((self.X, np.hstack((observations, actions))[:-1]))
			self.Y = np.vstack((self.Y, observations[1:]))

	def train(self):
		self.scaler = StandardScaler()
		self.scaler.fit(self.X)
		X_train = self.scaler.transform(self.X)
		self.model = MLPRegressor(hidden_layer_sizes=(32,32))
		self.model.fit(X_train, self.Y)

	def predict(self, observation, action):
		obs = self.scaler.transform(np.asarray([*observation, action]).reshape(1,-1))
		y_pred = self.model.predict(obs)
		return y_pred.flatten(), 1.0, 0.25<np.abs(y_pred[0,1]) or 1.0<np.abs(y_pred[0,0])

	def rollout(self, observation, policy, n):
		trajectory_pred = np.empty(n, self.output_dim)
		current_state = observation
		current_action = policy(state)
		for i in range(n):
			trajectory_pred[i] = self.predict(current_state, current_action)
			current_state = trajectory_pred[i]
			current_action = policy(current_state)
		return trajectory_pred, actions
