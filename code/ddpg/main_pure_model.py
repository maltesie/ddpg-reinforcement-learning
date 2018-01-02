#Implementation of Deep Deterministic Gradient with Tensor Flow"
# Author: Steven Spielberg Pon Kumar (github.com/stevenpjg)

import gym
from gym.spaces import Box, Discrete
import numpy as np
from ddpg import DDPG
from ou_noise import OUNoise
from gp import GP
#specify parameters here:
episodes=10000
is_batch_norm = False #batch normalization switch

def main():
    experiment= 'InvertedPendulum-v1' #specify environments here
    env = gym.make(experiment)
    steps= env.spec.timestep_limit #steps per episode  
    assert isinstance(env.observation_space, Box), "observation space must be continuous"
    assert isinstance(env.action_space, Box), "action space must be continuous"
    
    #Randomly initialize critic,actor,target critic, target actor network  and replay buffer   
    agent = DDPG(env, is_batch_norm)
    exploration_noise = OUNoise(env.action_space.shape[0])
    eval_limit = 10
    counter=0
    reward_per_episode = 0    
    total_reward=0
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]    
    print("Number of States:", num_states)
    print("Number of Actions:", num_actions)
    print("Number of Steps per episode:", steps)
    #saving reward:
    reward_st = np.array([0])
    reward_eval_st = np.array([0])
    model = GP(num_states)
    
    for i in range(20):
        actions = []
        observations = []
        rewards = []
        observation = env.reset()
        exploration_noise.reset()
        for t in range(steps): 
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()
            action = action[0] + noise
            observations.append(observation)
            actions.append(action)
            observation,reward,done,info=env.step(action)
            if done: break
        model.add_trajectory(observations, actions)
    model.train()

    for i in range(steps):
        observation = env.reset()
        reward_per_episode = 0

        for t in range(steps):         
            x = observation
            action = agent.evaluate_actor(np.reshape(x,[1,num_states]))
            noise = exploration_noise.noise()
            action = action[0] + noise
            observations.append(observation)
            actions.append(action)
            observation,reward,done = model.predict(x, action)
            agent.add_experience(x,observation,action,reward,done)

            if counter > 64: 
                agent.train()
            reward_per_episode+=reward
            counter+=1

            if (done or (t == steps-1)):
                print( 'EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_per_episode)
                exploration_noise.reset() #reinitializing random noise for action exploration
                reward_st = np.append(reward_st,reward_per_episode)
                np.savetxt('episode_reward_model.txt',reward_st, newline="\n")
                break
        
        if i % 4 == 0:
            observations = []
            actions = []
            observation = env.reset()
            reward_per_episode = 0
            reward_eval = 0
            for t in range(steps):      
                env.render()   
                observations.append(observation)
                action = agent.evaluate_actor(np.reshape(observation,[1,num_states]))[0]
                actions.append(action)
                observation, reward, done, info = env.step(action)
                reward_eval += reward
                if done: break
            if reward_eval > eval_limit:
                eval_limit += 5
                model.add_trajectory(observations[-8:], actions[-8:])
                model.train()
            print( 'EVALUATION: EPISODE: ',i,' Steps: ',t,' Total Reward: ',reward_eval)
            reward_eval_st = np.append(reward_eval_st,reward_eval)
            np.savetxt('episode_reward_model_eval.txt',reward_eval_st, newline="\n")            
    
    total_reward+=reward_per_episode            
    print("Average reward per episode {}".format(total_reward / episodes))


if __name__ == '__main__':
    main()    