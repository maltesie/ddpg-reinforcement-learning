import numpy as np
import matplotlib.pyplot as plt

evaluation = np.loadtxt('results/run_.,tag_Evaluation.csv', delimiter=',', skiprows=1)
reward = np.loadtxt('results/run_.,tag_Reward.csv', delimiter=',', skiprows=1)
plt.plot(reward[:,1], reward[:,2], 'r-', markersize=5, label=u'within model')
plt.plot(evaluation[:,1], evaluation[:,2],  'b-', markersize=5, label=u'in environment')
print(reward)
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend(loc='upper left')
plt.show()