import numpy as np


def done_cartpole(state):
    return 0.21<np.abs(state[0,2]) or 2.4<np.abs(state[0,0])

def done_pendulum(state):
    return False

def reward_cartpole(state, action):
    return 1.0

def reward_pendulum(state, action):
    state[np.abs(state)>1.0] = 1.0
    return -(np.arccos(state[0,0])**2 + 0.1*state[0,2]**2 + 0.001*action**2)
