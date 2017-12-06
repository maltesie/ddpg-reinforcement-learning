import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy

np.random.seed(1)

env = normalize(GymEnv("CartPole-v0"))

policy = CategoricalMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(32, 32)
    )

x = []
y = []
for i in range(5):
    observation = env.reset()
    done = False
    while not done:
        action, probs = policy.get_action(observation)
        x.append([action, *observation])
        
        observation, reward, done, info = env.step(action)
        y.append(observation)

x = np.array(x)
mean = x.mean(axis=0)
#x -= mean
#print(x)
y = np.array(y)

# Instanciate a Gaussian Process model
kernel = RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

gp.fit(x, y.reshape(-1,4))

x,y = [], []
observation = env.reset()
done = False
while not done:
    action, probs = policy.get_action(observation)
    x.append([action, *observation])
    
    observation, reward, done, info = env.step(action)
    y.append(observation)

x = np.array(x)
#x -= mean
#print(x)
y = np.array(y)

y_pred, sigma = gp.predict(x, return_std=True)

feature = 0
fig = plt.figure()
fig.clf()
plt.plot(x[:,feature+1], y[:,feature], 'r.', markersize=10, label=u'Observations')
plt.plot(x[:,feature+1], y_pred[:,feature], 'b-', label=u'Prediction')
plt.ylim(np.vstack((y[:,feature],y_pred[:,feature])).min(), np.vstack((y[:,feature],y_pred[:,feature])).max())
plt.legend(loc='upper left')
plt.show()

feature = 1
fig = plt.figure()
fig.clf()
plt.plot(x[:,feature+1], y[:,feature], 'r.', markersize=10, label=u'Observations')
plt.plot(x[:,feature+1], y_pred[:,feature], 'b-', label=u'Prediction')
plt.ylim(np.vstack((y[:,feature],y_pred[:,feature])).min(), np.vstack((y[:,feature],y_pred[:,feature])).max())
plt.legend(loc='upper left')
plt.show()

feature = 2
fig = plt.figure()
fig.clf()
plt.plot(x[:,feature+1], y[:,feature], 'r.', markersize=10, label=u'Observations')
plt.plot(x[:,feature+1], y_pred[:,feature], 'b-', label=u'Prediction')
plt.ylim(np.vstack((y[:,feature],y_pred[:,feature])).min(), np.vstack((y[:,feature],y_pred[:,feature])).max())
plt.legend(loc='upper left')
plt.show()

feature = 3
fig = plt.figure()
fig.clf()
plt.plot(x[:,feature+1], y[:,feature], 'r.', markersize=10, label=u'Observations')
plt.plot(x[:,feature+1], y_pred[:,feature], 'b-', label=u'Prediction')
plt.ylim(np.vstack((y[:,feature],y_pred[:,feature])).min(), np.vstack((y[:,feature],y_pred[:,feature])).max())
plt.legend(loc='upper left')
plt.show()