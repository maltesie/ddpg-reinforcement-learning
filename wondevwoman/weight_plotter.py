#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt


colors = ['#0055AA', '#CC0022', '#DDB800']


def plot(filename, nb_curves):
	layer_weights_history = None
	with open(filename, 'r') as f:
		f.readline() # skip shape
		for line in f:
			layers = line.split('; ')

			if layer_weights_history is None:
				layer_weights_history = [[] for _ in range(len(layers))]

			for i, layer in enumerate(layers):
				weights = np.array(list(map(float, layer.split(', '))))
				#print(weights)
				#exit(1)
				layer_weights_history[i].append(weights)

	for layer_idx, weights_history in enumerate(layer_weights_history):
		weights_history = np.array(weights_history)
		weight_indices = np.random.choice(np.arange(weights_history.shape[1]), nb_curves, replace=False)
		for i in weight_indices:
			plt.plot(weights_history[:, i], color=colors[layer_idx])
	plt.show()


if __name__ == '__main__':
	if len(sys.argv) != 3:
		print('Usage: %s <filename> <number of curves per layer>' % sys.argv[0])
		exit(1)
	filename = sys.argv[1]
	nb_curves = int(sys.argv[2])
	plot(filename, nb_curves)
