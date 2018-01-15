#!/usr/bin/env python3

'''plots multiple evaluation score (eval-scores*.txt) files to compare parameter sets'''

import sys
import numpy as np
import matplotlib.pyplot as plt


colors = ['#0055AA', '#CC0022', '#DDB800']


def plot(filenames, step_size):
    '''reads files speciefied by `filenames` and plots them on an axis with speciefied `step_size`'''
    for i, filename in enumerate(filenames):
        # gather score series
        scoress = []
        with open(filename, 'r') as f:
            for line in f:
                scores = line[1:-2].split(', ')
                scoress.append(scores)
        scoress = np.asarray(scoress, dtype=np.float64)
        # aggregate
        scores = np.mean(scoress, axis=0)
        stderr = scoress.std(axis=0, ddof=1) / np.sqrt(scoress.shape[0])
        # create x axis
        X = range(step_size, (scoress.shape[1] + 1) * step_size, step_size)
        # strip away bloating filename parts for label
        label = filename.replace('eval-scores-', '').replace('.txt', '')
        # plot!
        plt.plot(X, scores, color=colors[i], label=label)
        plt.errorbar(X, scores, stderr, color=colors[i])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: %s <filenames ...> <x tick step size>' % sys.argv[0])
        exit(1)
    filenames = sys.argv[1:-1]
    step_size = int(sys.argv[-1])
    plot(filenames, step_size)
