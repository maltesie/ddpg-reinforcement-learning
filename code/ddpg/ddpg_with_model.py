# ================================================
# Modified from the work of Patrick Emami:
#       Implementation of DDPG - Deep Deterministic Policy Gradient
#       Algorithm and hyperparameter details can be found here:
#           http://arxiv.org/pdf/1509.02971v2.pdf
#
# Removed TFLearn dependency
# Added Ornstein Uhlenbeck noise function
# Added reward discounting
# Works with discrete actions spaces (Cartpole)
# Tested on CartPole-v0 & -v1 & Pendulum-v0
# Author: Liam Pettigrew
# =================================================
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gym

from replay_buffer import ReplayBuffer
from noise import Noise
from reward import Reward
from actor import ActorNetwork
from critic import CriticNetwork

from gp import GP
from nn import NN
from actionsampler import ActionSampler
from functions import done_pendulum, reward_pendulum, done_cartpole, reward_cartpole


# ==========================
#   Meta Parameters
# ==========================

session_prefix = 'average-gp-model-5-eval'
nb_sessions = 5


# ==========================
#   Model Parameters
# ==========================

# Toggle model use
use_model = True
# Toggle exploration noise
use_noise = True
#Model architecture: GP or NN
M = GP
# Model pretraining episodes
model_train_ep = 2
# Model pretraining steps per episode
model_train_steps = 100
# Number of samples the model uses
nb_samples = 500
# Interval of evaluation and retraining
nb_ep_eval = 5

# ==========================
#   Training Parameters
# ==========================

# Maximum episodes run
MAX_EPISODES = 1000
# Episodes with noise
NOISE_MAX_EP = 1000
# Noise parameters - Ornstein Uhlenbeck
if use_model: noise_factor = 0.6
else: noise_factor = 0.6
DELTA = 0.5 # The rate of change (time)
SIGMA = 0.5 # Volatility of the stochastic processes
OU_A = 3. # The rate of mean reversion
OU_MU = 0. # The long run average interest rate
# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================

#ENV_NAME = 'CartPole-v0' # Discrete: Reward factor = 0.01
#ENV_NAME = 'CartPole-v1' # Discrete: Reward factor = 0.01
ENV_NAME = 'Pendulum-v0' # Continuous: Reward factor = 0.001
# Max episode length and reward factor
if ENV_NAME == 'Pendulum-v0': 
    MAX_EP_STEPS = 200
    REWARD_FACTOR = 0.001
elif ENV_NAME == 'CartPole-v1': 
    MAX_EP_STEPS = 500
    REWARD_FACTOR = 0.01
elif ENV_NAME == 'CartPole-v0': 
    MAX_EP_STEPS = 200
    REWARD_FACTOR = 0.01
# Directory for storing gym results
MONITOR_DIR = './results/' + ENV_NAME
# Directory for storing tensorboard summary results
SUMMARY_DIR = './results/' + ENV_NAME + '/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer
BUFFER_SIZE = 100000
MINIBATCH_SIZE = 100
# Toggle model error plot
plot_mse = False
# Toggle predict change
predict_change = False
# Render gym env during training
RENDER_ENV = False
# Use Gym Monitor
GYM_MONITOR_EN = False
s_nb = 0
plot_data = []

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax", episode_ave_max_q)
    max_mse = tf.Variable(0.)
    tf.summary.scalar("mse", max_mse)
    summary_vars = [episode_reward, episode_ave_max_q, max_mse]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(sess, env, model, actor, critic, noise, reward, discrete):
    # Set up summary writer
    summary_writer = tf.summary.FileWriter(SUMMARY_DIR)

    sess.run(tf.global_variables_initializer())

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

    # Initialize noise
    ou_level = 0.
    
    for i in range(MAX_EPISODES):
        
        ep_reward = 0
        ep_ave_max_q = 0
        
        if i % nb_ep_eval == 0 and use_model:
            episode_buffer = np.empty((0,5), float)
            observations = [env.reset()]
            actions = []
            rewards_eval = []
            for t in range(MAX_EP_STEPS):      
                env.render()   
                a = actor.predict(np.reshape(observations[-1], (1, actor.s_dim)))
                if i < NOISE_MAX_EP and use_noise:
                    ou_level = noise.ornstein_uhlenbeck_level(ou_level)
                    a = a + noise_factor * ou_level
                if discrete:
                    action = np.argmax(np.abs(a))
                else:
                    action = a[0]
                
                actions.append(action)
                observation, r, done, info = env.step(action)
                
                observations.append(observation)
                rewards_eval.append(r)
                
                episode_buffer = np.append(episode_buffer, [[observations[-2], a, r, done, observations[-1]]], axis=0)
                
                if done or t==MAX_EP_STEPS-1:
                    rewards_eval = np.asarray(rewards_eval).reshape(-1,1)
                    model.add_trajectory(np.asarray(observations), np.asarray(actions).reshape(-1,1), rewards_eval)
                    model.train()
                    
                    episode_buffer = reward.discount(episode_buffer)
                    plot_data.append((rewards_eval.sum(), 0.))
                    for step in episode_buffer:
                        replay_buffer.add(np.reshape(step[0], (actor.s_dim,)), np.reshape(step[1], (actor.a_dim,)), step[2], \
                                  step[3], np.reshape(step[4], (actor.s_dim,)))
                    break
            print( '| EVALUATION | Reward: ', rewards_eval.sum())
        # Clear episode buffer
        episode_buffer = np.empty((0,5), float)
        s = env.reset()  
        predicted_state = s
        mses = []
        for j in range(MAX_EP_STEPS):
            if RENDER_ENV:
                env.render()
            a = actor.predict(np.reshape(s, (1, actor.s_dim)))

            # Add exploration noise
            if i < NOISE_MAX_EP and use_noise:
                ou_level = noise.ornstein_uhlenbeck_level(ou_level)
                a = a + noise_factor * ou_level

            # Set action for discrete and continuous action spaces
            if discrete:
                action = np.argmax(np.abs(a))
            else:
                action = a[0]
                
            if not use_model:
                s2, r, terminal, info = env.step(action)
                predicted_state, predicted_r, _ = model.predict(predicted_state,action)
                mses.append(np.sqrt(np.sum((np.asarray(s2)-np.asarray(predicted_state))**2)+(r-predicted_r)**2))
            else:
                s2, r, terminal = model.predict(s, action)
                actual_state, actual_r, _, _ = env.step(action)
                mses.append(np.sqrt(np.sum((np.asarray(s2)-np.asarray(actual_state))**2)+(r-actual_r)**2))
            # Choose reward type
            ep_reward += r

            episode_buffer = np.append(episode_buffer, [[s, a, r, terminal, s2]], axis=0)

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > MINIBATCH_SIZE:
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(MINIBATCH_SIZE)

                # Calculate targets
                target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(MINIBATCH_SIZE):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + GAMMA * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            # Set previous state for next step
            s = s2

            if (terminal or j==MAX_EP_STEPS-1):
                # Reward system for episode
                #episode_buffer = reward.total(episode_buffer, ep_reward)
                episode_buffer = reward.discount(episode_buffer)

                # Add episode to replay buffer
                for step in episode_buffer:
                    replay_buffer.add(np.reshape(step[0], (actor.s_dim,)), np.reshape(step[1], (actor.a_dim,)), step[2], \
                                  step[3], np.reshape(step[4], (actor.s_dim,)))
                
                plot_data.append((ep_reward, np.mean(mses)))
                
                summary = tf.Summary()
                summary.value.add(tag='Reward', simple_value=float(ep_reward))
                summary.value.add(tag='Qmax', simple_value=float(ep_ave_max_q / float(j)))
                summary.value.add(tag='mse', simple_value=float(np.mean(mses)))
                summary_writer.add_summary(summary, i)

                summary_writer.flush()
                print('| Reward: %.2i' % int(ep_reward), '| Steps: %.2i' % int(j), " | Episode", i, \
                '| Qmax: %.4f' % (ep_ave_max_q / float(j)), '| Modelerror: ', np.mean(mses))

                break


def main(_):
    with tf.Session() as sess:
        env = gym.make(ENV_NAME)
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        print(env.observation_space)
        print(env.action_space)

        state_dim = env.observation_space.shape[0]

        try:
            action_dim = env.action_space.shape[0]
            action_bound = env.action_space.high
            # Ensure action bound is symmetric
            assert (env.action_space.high == -env.action_space.low)
            discrete = False
            print('Continuous Action Space')
        except (ValueError, IndexError):
            action_dim = env.action_space.n
            action_bound = 1
            discrete = True
            print('Discrete Action Space')

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             ACTOR_LEARNING_RATE, TAU)
        
        critic = CriticNetwork(sess, state_dim, action_dim,
                               CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        noise = Noise(DELTA, SIGMA, OU_A, OU_MU)
        reward = Reward(REWARD_FACTOR, GAMMA)
        
        if ENV_NAME == 'Pendulum-v0':
            model = M(state_dim, done_pendulum, predict_change=predict_change)
            sampler = ActionSampler(False, action_bounds=[-2.0,2.0])
        elif ENV_NAME == 'CartPole-v0':
            model = M(state_dim, done_cartpole, predict_change=predict_change)
            sampler = ActionSampler(True, actions=[0,1])
        elif ENV_NAME == 'CartPole-v1':
            model = M(state_dim, done_cartpole, predict_change=predict_change)
            sampler = ActionSampler(True, actions=[0,1])

        for i in range(model_train_ep):
            observations = [env.reset()]
            actions = sampler.sample(model_train_steps)
            rs = []
            #print(actions)
            for t in range(model_train_steps): 
                observation,r,ddd,_ = env.step(actions[t])
                #if ddd: break
                observations.append(observation)
                rs.append(r)
            rs = np.asarray(rs)
            observations = np.asarray(observations)
            model.add_trajectory(observations, actions.reshape(-1,1), rs.reshape(-1,1))
        model.train()

        if GYM_MONITOR_EN:
            if not RENDER_ENV:
                env = Monitor(env, MONITOR_DIR, video_callable=False, force=True)
            else:
                env = Monitor(env, MONITOR_DIR, force=True)   
                
        try:
            train(sess, env, model, actor, critic, noise, reward, discrete)
        except KeyboardInterrupt:
            pass
        plot_results(plot_data, model, ENV_NAME, use_model, use_noise, model_train_ep, model_train_steps, nb_samples, nb_ep_eval)
        with open('./results/' + ENV_NAME + '/np_dumps/{}_{}.npy'.format(session_prefix, str(s_nb)), 'wb') as file:
            np.save(file, np.asarray(plot_data))
        if GYM_MONITOR_EN:
            env.close()

def plot_results(plot_data, model, env_name, use_model, use_noise, model_train_ep, model_train_steps, nb_samples, nb_ep_eval):
    plot_data = np.asarray(plot_data)
    fig, ax1 = plt.subplots()
    ax1.plot(plot_data[:,0], 'r-', markersize=5, label=u'within model')
    ax1.plot(plot_data[:,1],  'b-', markersize=5, label=u'in environment')
    ax1.set_xlabel('episode')
    ax1.set_ylabel('reward')
    ax1.legend(loc='upper left')
    if plot_mse:
        ax2 = ax1.twinx()
        ax2.plot(plot_data[:,2], 'g-', markersize=5, label=u'model mse')
        ax2.set_ylabel('max. mse', color='g')
        ax2.legend(loc='up')
    if use_model:
        if use_noise: noise = 'Used exploration noise.' 
        else: noise = 'Without exploration noise.'
        title = 'Training in environment {0} with {1} model.\nPretrained with {2}*{3} samples and retrained every {4} episodes\nwith {5} samples. {6}'.format(
        env_name, model.type, model_train_ep, model_train_steps, nb_ep_eval, nb_samples, noise)
    else:
        title = 'Training in environment {0} without model'.format(env_name)
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    with tf.device('/device:CPU:0'):
        for nb_sessions in [5, 10]:
            session_prefix = 'average-gp-model-{}-eval'.format(nb_sessions)
            for session_nb in range(nb_sessions):
                s_nb = session_nb
                plot_data = []
                try:
                    tf.app.run()
                except SystemExit:
                    pass
    