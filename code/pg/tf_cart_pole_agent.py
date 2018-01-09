'''simple policy gradient reinforcement learning on opengym's cart pole task using tensorflow'''

import tensorflow as tf
import numpy as np
import gym
import time

def get_discounted_rewards(rewards):
    '''adds up each reward and its following rewards with a discount factor. centers and normalizes.'''
    gamma = 0.99 # reward back-through-time aggregation ratio
    r = np.asarray(rewards, dtype=np.float64)

    if np.all(r == 1):
        return None

    for i in range(len(r) - 2, -1, -1):
        r[i] = r[i] + gamma * r[i+1]

    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    r -= np.mean(r)
    r /= np.std(r)

    return r


# neural net setup:
# x -> W1 -> W2 -> ap
#       |
#       v
# a -> W3 -> nx

# observation input and trainable weights
net_xs = tf.placeholder(tf.float32, [None, 4])
net_W1 = tf.get_variable("W1", shape=[4, 8], initializer=tf.contrib.layers.xavier_initializer())
net_W2 = tf.get_variable("W2", shape=[8, 1], initializer=tf.contrib.layers.xavier_initializer())
net_W3 = tf.get_variable("W3", shape=[9, 4], initializer=tf.contrib.layers.xavier_initializer())

# backprop placeholders
net_rewards = tf.placeholder(tf.float32, [None])
net_actions = tf.placeholder(tf.float32, [None])
net_nxs = tf.placeholder(tf.float32, [None, 4])

# intermediate layer "world features"
net_world_features = tf.nn.leaky_relu(tf.matmul(net_xs, net_W1), alpha=0.1)

# output: action probabilities
net_aps = tf.nn.sigmoid(tf.matmul(net_world_features, net_W2))

# output: next x estimates
net_nxes = tf.matmul(tf.concat([net_world_features, tf.expand_dims(net_actions, axis=1)], axis=1), net_W3)

# loss functions of the outputs above
loss_actions = -tf.reduce_mean(net_rewards * tf.log(tf.multiply(1 - net_actions, 1 - net_aps[:, 0]) + tf.multiply(net_actions, net_aps[:, 0])))
loss_nxs = tf.reduce_mean((net_nxs - net_nxes) ** 2)

# training methods
train_policy = tf.train.RMSPropOptimizer(learning_rate=.01, decay=.99).minimize(loss_actions)
train_model = tf.train.RMSPropOptimizer(learning_rate=.01, decay=.99).minimize(loss_nxs)

net_session = tf.InteractiveSession()
tf.global_variables_initializer().run()

env = gym.make('CartPole-v1')

running_mean = 22. # init with rougly the average step count of a random agent

for i in range(10000): # terminate eventually
    # episode log
    steps = 0
    actions = []
    rewards = []
    xs = []
    nxs = []

    # let's go
    x = env.reset()
    done = False
    while not done:
        #env.render()
        #time.sleep(0.075)
        xs.append(x)
        y = net_aps.eval(feed_dict={net_xs: [x]})[0][0] # get action
        action = 0 if np.random.random() > y else 1
        actions.append(action)
        x, reward, done, info = env.step(action)
        rewards.append(-1. if done else 1.)
        nxs.append(x)
        steps += 1

    running_mean = .99 * running_mean + .01 * steps
    print('\rcurrent step count estimate: %.2f' % running_mean, end='', flush=True)

    # train policy
    discounted_rewards = get_discounted_rewards(rewards)
    net_session.run(train_policy, feed_dict={net_xs: xs, net_rewards: discounted_rewards, net_actions: actions})

    # train model
    net_session.run(train_model, feed_dict={net_xs: xs, net_nxs: nxs, net_actions: actions})
