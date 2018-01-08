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


# neural net setup
net_xs = tf.placeholder(tf.float32, [None, 4])
net_W1 = tf.get_variable("W1", shape=[4, 8], initializer=tf.contrib.layers.xavier_initializer())
net_W2 = tf.get_variable("W2", shape=[8, 1], initializer=tf.contrib.layers.xavier_initializer())
net_ys = tf.nn.sigmoid(tf.matmul(tf.nn.leaky_relu(tf.matmul(net_xs, net_W1), alpha=0.1), net_W2))

# backprop inputs
net_rewards = tf.placeholder(tf.float32, [None])
net_actions = tf.placeholder(tf.float32, [None])

# L = reward * ((1 - action) - (p * (1 - 2 * action)))
loss = -tf.reduce_mean(net_rewards * tf.log(tf.multiply(1 - net_actions, 1 - net_ys[:, 0]) + tf.multiply(net_actions, net_ys[:, 0])))

#train_step = tf.train.GradientDescentOptimizer(.05).minimize(loss)
train_step = tf.train.RMSPropOptimizer(learning_rate=.01, decay=.99).minimize(loss)

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

    # let's go
    x = env.reset()
    done = False
    while not done:
        #env.render()
        #time.sleep(0.075)
        xs.append(x)
        y = net_ys.eval(feed_dict={net_xs: [x]})[0][0] # get action
        action = 0 if np.random.random() > y else 1
        actions.append(action)
        x, reward, done, info = env.step(action)
        rewards.append(-1. if done else 1.)
        steps += 1

    running_mean = .99 * running_mean + .01 * steps
    print('\rcurrent step count estimate: %.2f' % running_mean, end='', flush=True)
    discounted_rewards = get_discounted_rewards(rewards)
    net_session.run(train_step, feed_dict={net_xs: xs, net_rewards: discounted_rewards, net_actions: actions})
