'''script to generate cart pole trajectories with grid-like distributed starting configurations'''

import gym
import numpy as np
import json, sys

def count(counters, max_digit):
	'''basically a +1 operation with carry for an array'''
	for i in range(len(counters) - 1, -1, -1):
		counters[i] += 1
		if counters[i] > max_digit:
			counters[i] = 0
		else:
			break
	return

def log_state(state, trajectory):
	'''appends `state` to `trajectory` with proper formatting'''
	trajectory.append(list(map(lambda a: round(a, 5), state)))

env = gym.make('CartPole-v1').unwrapped # the unwrapped variant allows state tinkering

# starting configuration space
low = np.asarray([-.05, -.05, -.05, -.05])
high = np.asarray([.05, .05, .05, .05])

# iterations per dimension
try:
	samples_per_dimension = int(sys.argv[1])
except Exception as e:
	samples_per_dimension = 4 # 4^4 = 256 trajectories

# where to store the trajectories
try:
	filename = sys.argv[2]
except Exception as e:
	filename = 'cartpole-trajectories.txt'

counters = np.asarray([0, 0, 0, 0]) # one per dimension
max_counters = np.asarray([samples_per_dimension - 1] * len(low))

nb_trajectories = 0

while True:
	x = env.reset()
	initial_state = (high - low) / (max_counters) * counters + low
	env.state = initial_state

	trajectory = []
	log_state(initial_state, trajectory)

	done = False
	while not done:
		# pick random action
		action = 0 if np.random.random() < 0.5 else 1
		trajectory.append(action)
		# progress environment
		x, reward, done, info = env.step(action)
		log_state(x, trajectory)

	# overwrite file on first time, then append
	with open(filename, 'w' if np.all(counters == 0) else 'a') as f:
		json.dump(trajectory, f)
		f.write('\n')
	nb_trajectories += 1

	count(counters, samples_per_dimension - 1)
	# end when all counters are back on the initital configuration
	if np.all(counters == 0):
		break

print('generated %i trajectories' % nb_trajectories)
