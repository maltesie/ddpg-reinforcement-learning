import os
import numpy as np


class WeightLogger(object):
	'''logging module for net weights'''
	def __init__(self, filename, overwrite=False, nb_weights=None):
		'''gets a log `filename` and optionally whether to overwrite an existing one and how many weights to store per layer (sparse logging)'''
		self.filename = filename
		if not overwrite and os.path.exists(filename):
			raise RuntimeError('The log file %s already exists.' % filename)
		self.shapes = None
		# for sparse logging
		self.nb_weights = nb_weights
		if nb_weights is None:
			self.sparse = False
		else:
			self.sparse = True
			self.layer_weight_indices = None

	def log_shapes(self, layers):
		'''stores the layer's shapes'''
		with open(self.filename, 'w') as f:
			f.write('; '.join(map(lambda layer: ', '.join(map(str, layer.shape)), layers)) + '\n')
		self.shapes = [layer.shape for layer in layers]

	def log(self, layers):
		'''stores list of weight layers'''
		# always store shapes as first line
		if self.shapes is None:
			self.log_shapes(layers)

		# initialize indices for sparse logging if needed
		if self.sparse:
			if self.layer_weight_indices is None:
				self.layer_weight_indices = [np.random.choice(np.arange(len(layer.ravel())), self.nb_weights, replace=False) for layer in layers]
			layer_weights = [layer.ravel()[indices] for layer, indices in zip(layers, self.layer_weight_indices)]
		else:
			layer_weights = [layer.ravel() for layer in layers]

		# log!
		with open(self.filename, 'a') as f:
			f.write('; '.join(map(lambda weights: ', '.join(map(lambda f: '%.6f' % f, weights)), layer_weights)) + '\n')
