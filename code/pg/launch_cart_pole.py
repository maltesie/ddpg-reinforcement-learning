import gym
import numpy as np
import time, sys

from np_cart_pole_agent import Agent


if len(sys.argv) > 1:
    mode = sys.argv[1]
    if not mode in ['demo', 'evaluate']:
        print('usage: ' + sys.argv[0] + ' [demo|evaluate]')
        sys.exit(1)
else:
    mode = 'evaluate' # 'demo', 'evaluate'

env = gym.make('CartPole-v1')

# hyperparameters
param = {'learn_model': True, 'rect_leakiness': 0.1}


if mode == 'demo':
    agent = Agent(**param)
    try:
        # not the actual batch training size, only for statistics and status updates
        batch_size = 100
        while True:
            nb_steps = agent.train_games(env, batch_size)
            avg_nb_steps = sum(nb_steps) / float(batch_size)
            max_steps = max(nb_steps)
            print('\rgame #%i avg number of steps: %.2f (max: %i)' % (agent.nb_games, avg_nb_steps, max_steps), end='', flush=True)

    except KeyboardInterrupt:
        print('\nTrained %i games. Showtime!' % agent.nb_games)

        observation = env.reset()
        done = False
        while not done:
            env.render()

            action = agent.get_action(observation, training=False)
            observation, reward, done, info = env.step(action)
            time.sleep(0.075)

elif mode == 'evaluate':
    import matplotlib.pyplot as plt

    # instances to train and average
    nb_trainings = 100
    # number of games to train on
    nb_games = 2000
    # steps size (in games) at which to perform an evaluation
    batch_size = 100
    # instances to evaluate and average
    nb_evaluations = 10

    scoress = [[] for _ in range(nb_trainings)]
    for t in range(nb_trainings):
        print('\rtrainging instance %i/%i' % (t + 1, nb_trainings))
        agent = Agent(**param, log_filename=False)
        while agent.nb_games < nb_games:
            print('\rgame %i/%i' % (agent.nb_games, nb_games), end='', flush=True)
            agent.train_games(env, batch_size)
            avg_steps_achieved = agent.evaluate_games(env, nb_evaluations)
            scoress[t].append(avg_steps_achieved)
    # include hyperparameters in filename where to store the individual scores of each training run
    filename = 'eval-scores-%s.txt' % '.'.join([str(k) + '-' + str(v) for k, v in param.items()])
    with open(filename, 'w') as f:
        [f.write(str(scores) + '\n') for scores in scoress]
    # compute mean of scores across training runs and standard error of the mean
    scoress = np.array(scoress, dtype=np.float64)
    scores = np.mean(scoress, axis=0)
    stderr = scoress.std(axis=0, ddof=1) / np.sqrt(scoress.shape[0])
    print()
    print(param)
    print('scores:', list(scores))
    print('stderr:', list(stderr))
    X = range(batch_size, nb_games + batch_size, batch_size)
    plt.plot(X, scores)
    plt.errorbar(X, scores, stderr)
    plt.show()
