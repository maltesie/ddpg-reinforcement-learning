import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time, sys, json

import tf_cart_pole_agent
from plot_eval import colors

def read_trajectories(filename, test_denom=4):
    '''reads cartpole trajectories and returns them as (xs, actions, nxs) tuples for training set and test set each.
       1 / `test_denom` of data points is returned as test set, so default is 25% test, 75% train'''
    train = [[], [], []]
    test = [[], [], []]
    with open(filename) as f:
        for line_nb, line in enumerate(f):
            trajectory = json.loads(line)
            for i in range(0, len(trajectory) - 2, 2):
                if line_nb % test_denom == 0:
                    test[0].append(trajectory[i])
                    test[1].append(trajectory[i + 1])
                    test[2].append(trajectory[i + 2])
                else:
                    train[0].append(trajectory[i])
                    train[1].append(trajectory[i + 1])
                    train[2].append(trajectory[i + 2])
    return train, test

def train_model(agent, train_set, test_set, batch_size=100, episodes=1000):
    '''samples `batch_size` data points each to do `episodes` model training steps.
       returns model error over time.'''
    agent.experience['xs'], agent.experience['actions'], agent.experience['nxs'] = train_set

    errors = [agent.get_model_error(*test_set)]

    for i in range(episodes):
        xs, actions, dxs = agent.sample_experience(batch_size, agent.model_training_noise)
        agent.net_session.run(agent.train_model, feed_dict={
            agent.net_xs: xs,
            agent.net_actions: actions,
            agent.net_dxs: dxs})

        agent.model_training_noise *= agent.model_training_noise_decay

        err = agent.get_model_error(*test_set)
        errors.append(err)

        if i % 100 == 0:
            print('\r%i / %i' % (i, episodes), end='', flush=True)

    print()

    return errors


if __name__ == '__main__':
    # param_sets = [{'model_training_noise': 0},
    #         {'model_training_noise': 1.0},
    #         {'model_training_noise': 2.0},
    #         {'model_training_noise': 3.0}]

    # param_sets = [{'model_training_noise': 0},
    #         {'model_training_noise': 3.0},
    #         {'model_training_noise': 5.0},
    #         {'model_training_noise': 10., 'model_training_noise_decay': 0.99}]

    # lrelu = lambda x: tf.nn.leaky_relu(x, alpha=0.01)
    # param_sets = [{'model_training_noise': 2.0, 'model_training_noise_decay': 1.0, 'model_afunc': tf.nn.relu, 'learn_model': 'delta'},
    #         {'model_training_noise': 2.0, 'model_training_noise_decay': 1.0, 'model_afunc': tf.nn.relu, 'learn_model': 'absolute'}]

    # param_sets = [{'learn_model': 'delta', 'rect_leakiness': 0.0},
    #         {'learn_model': 'delta', 'rect_leakiness': 0.2},
    #         {'learn_model': 'delta', 'rect_leakiness': 1.0},
    #         {'learn_model': 'delta', 'rect_leakiness': 1.5}]

    param_sets = [{'model_afunc': tf.nn.relu},
            {'model_afunc': tf.nn.sigmoid},
            {'model_afunc': tf.nn.tanh},
            {'model_afunc': None}]

    try:
        trajectories_filename = sys.argv[1]
    except Exception as e:
        trajectories_filename = 'cartpole-trajectories.txt'

    # over how many runs to average per param set
    iterations = 10

    train, test = read_trajectories(trajectories_filename, 5)
    print('train set: %i, test set: %i' % (len(train[0]), len(test[0])))

    for i, params in enumerate(param_sets):
        print('parameters:', params)
        errors = []
        for _ in range(iterations):
            agent = tf_cart_pole_agent.Agent(**params, random_seed=31415926+iterations)
            errors.append(train_model(agent, train, test))

        errors = np.asarray(errors)
        stderr = errors.std(axis=0, ddof=1) / np.sqrt(errors.shape[0])
        mean_error = np.mean(errors, axis=0)

        label = ' '.join([str(k) + '=' + str(v) for k, v in sorted(params.items())])
        #label = 'η=%.1f' % params['model_training_noise'] + ((' γ=%f' % params['model_training_noise_decay']).rstrip('0') if 'model_training_noise_decay' in params else '')
        #label = params['learn_model']
        #label = ('α=%.2f' % params['rect_leakiness']).rstrip('0')
        plt.plot(mean_error, label=label, color=colors[i])
        #plt.errorbar(range(mean_error.shape[0]), mean_error, stderr, color=colors[i])

    plt.legend()
    plt.xlabel('number of trainings')
    plt.ylabel('model error')
    plt.show()
