import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class GP(object):
	"""docstring for GP"""
	def __init__(self, initial_state):
		self.initial_state = np.asarray(initial_state).reshape(1,-1)
		self.input_dim = len(initial_state)
		self.output_dim = self.input_dim -1
		self.X = None
		self.Y = None

	def add_trajectory(self, observations, actions):
		if not self.X:
			self.X = np.hstack((observations, actions))[:-1]
			self.Y = np.asarray(observations[1:])
		else:
			self.X = np.vstack((self.X, np.hstack((observations, actions))[:-1]))
			self.Y = np.vstack((self.Y, observations[1:]))

	def train(self):
		kernel = RBF(10, (1e-2, 1e2))
		self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
		self.model.fit(self.X, self.Y)

	def predict(self, observation, action, return_std=False):
		y_pred, sigma = gp.predict(np.asarray([*observation, action]), return_std=True)
		if return_std: return y_pred, sigma
		else: return y_pred

	def rollout(self, observation, policy, n):
		trajectory_pred = np.empty(n, self.output_dim)
		current_state = observation
		current_action = policy(state)
		for i in range(n):
			trajectory_pred[i] = self.predict(current_state, current_action)
			current_state = trajectory_pred[i]
			current_action = policy(current_state)
		return trajectory_pred, actions

