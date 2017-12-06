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
for i in range(20):
    observation = env.reset()
    done = False
    while not done:
        print(observation)
        action, probs = policy.get_action(observation)
        x.append([action, *observation])
        
        observation, reward, done, info = env.step(action)
        y.append(observation)

x = np.array(x)
x -= x.mean(axis=0)

y = np.array(y)
print(x.shape, len(y))

# Instanciate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

gp.fit(x, y.reshape(-1,1))
y_pred, sigma = gp.predict(x, return_std=True)

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
