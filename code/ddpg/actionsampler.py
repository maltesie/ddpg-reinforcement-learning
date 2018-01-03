import numpy as np

class ActionSampler(object):
    
    def __init__(self, discrete, action_bounds=None, actions=None):
        self.discrete = discrete
        if discrete:
            self.actions = np.asarray(actions)
        else:
            self.action_bounds = np.asarray(action_bounds)
            assert np.abs(self.action_bounds[0])==np.abs(self.action_bounds[1])
        
    def sample(self, n):
        if self.discrete:
            index=np.random.rand(n)
            index *= self.actions.size
            index = np.asarray(index, dtype=int)
            return self.actions[index]
        else:
            samples=(np.random.rand(n)-0.5)*2.0
            samples *= self.action_bounds[0]
            return samples