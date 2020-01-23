import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from random import uniform, randint
from collections import named tuple


Action = namedtuple('Action', ['position', 'action'])
Perspective = namedtuple('Perspective', ['perspective', 'position'])


class FooEnv(gym.Env):
metadata = {'render.modes': ['human']}
    

    def __init__(selfi, size):
        self.system_size = size
        self.plaquette_matrix = np.zeros((self.system_size, self.system_size), dtype=int)   # dont use self.plaquette
        self.vertex_matrix = np.zeros((self.system_size, self.system_size), dtype=int)      # dont use self.vertex 
        self.qubit_matrix = np.zeros((2, self.system_size, self.system_size), dtype=int)
        self.current_state = np.stack((self.vertex_matrix, self.plaquette_matrix,), axis=0)
        self.next_state = np.stack((self.vertex_matrix, self.plaquette_matrix), axis=0)
        self.ground_state = True    # True: only trivial loops, 
                                    # False: non trivial loop 

    def step(self, action):
        qubit_matrix = action.position[0]
        row = action.position[1]
        col = action.position[2]
        add_operator = action.action

        old_operator = self.qubit_matrix[qubit_matrix, row, col]
        new_operator = self.rule_table[int(old_operator), int(add_operator)]
        self.qubit_matrix[qubit_matrix, row, col] = new_operator        
        self.syndrom('next_state')    
        
    def reset(self):
        ...
    def render(self, mode='human'):
        ...
    def close(self):
        ...
