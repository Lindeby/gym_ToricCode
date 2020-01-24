import gym
import numpy as np
import matplotlib as plt
from collections import namedtuple

class ToricCodeEnv(gym.Env):
    """
    Description:

    Observation:

    Actions:
        the actions avaliable shuld be x, y and z operation on all
        position of the matrix
    Reward:

    Episode Termination:

    """

    Action = namedtuple('Action', ['position', 'action'])

    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, size=3, min_qbit_errors=0, p_error=0.1):
        self.system_size = size
        self.min_qbit_errors = min_qbit_errors
        self.p_error = p_error
        
        # plaquette_matrix and vertex_matrix combinded beckomes the state
        # that is avaliable for the agent
        self.plaquette_matrix   = np.zeros((self.system_size, self.system_size), dtype=int)   # dont use self.plaquette
        self.vertex_matrix      = np.zeros((self.system_size, self.system_size), dtype=int)      # dont use self.vertex 
        # qubit_matrix contains the true errors and shuld not be 
        # avaliable for for an agent
        self.qubit_matrix       = np.zeros((2, self.system_size, self.system_size), dtype=int)

        self.state              = np.stack((self.vertex_matrix, self.plaquette_matrix,), axis=0)
        self.next_state         = np.stack((self.vertex_matrix, self.plaquette_matrix), axis=0)

        self.rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)  # Identity = 0, pauli_x = 1, pauli_y = 2, pauli_z = 3

        self.ground_state = True    # True: only trivial loops, False: non trivial loop 
        # self.terminal_state = False


    def step(self, action):
        """Returns Observation(object), reward(float), done(bolean), info(dict)"""

        # Observation <-- currentstate
        
        # Reward <-- getReward

        # Done <-- Terminal State ?= 1 False ow True

        # info <-- nothing for now 
        
        qubit_matrix = action.position[0]
        row = action.position[1]
        col = action.position[2]
        add_operator = action.action

        old_operator = self.qubit_matrix[qubit_matrix, row, col]
        new_operator = self.rule_table[int(old_operator), int(add_operator)]
        self.qubit_matrix[qubit_matrix, row, col] = new_operator        
        self.syndrom('next_state')
        
        reward = self.get_reward()
        self.state = self.next_state                                                               #not sure about this previously done in training loop
        return self.state, reward, self.isTerminalState(self.state), {} 
       
        
    def reset(self):
        """Resets the environment and generates new errors.

        return - a reset state.
        """
        terminal_state = False

        qubit_matrix = np.zeros((2, self.system_size, self.system_size), dtype=int)

        # Generate new errors
        while terminal_state:
            if self.min_qbit_errors == 0:
                qubit_matrix = self.generateRandomError(qubit_matrix, self.p_error)
            else:
                qubit_matrix = self.generateNRandomErrors(qubit_matrix, self.min_qbit_errors)
            
            terminal_state = self.isTerminalState(qubit_matrix)

        self.qubit_matrix = qubit_matrix
        self.state = self.createSyndrom(self.qubit_matrix) # Create syndrom from errors

        return self.state


    def generateRandomError(self, matrix, p_error):
        """Generates errors with a probability.
        np matrix - a (2,x,x) numpy matrix the to generate errors on.
        double p_error - the probability to generate an error 
        
        return - the error matrix
        """
        for i in range(2):
            qubits = np.random.uniform(0, 1, size=(matrix.shape[1], matrix.shape[2]))
            error = qubits > p_error
            no_error = qubits < p_error
            qubits[error] = 0
            qubits[no_error] = 1
            pauli_error = np.random.randint(3, size=(matrix.shape[1], matrix.shape[2])) + 1
            matrix[i,:,:] = np.multiply(qubits, pauli_error)

        return matrix
    
    def generateNRandomErrors(self, matrix, n):
        """Generates n errors.
        np matrix - a (2,x,x) numpy matrix the to generate errors on.
        int n - the number of generated errors.

        return - the error matrix
        """ 
        errors = np.random.randint(3, size = n) + 1
        qubit_matrix_error = np.zeros(2*matrix.shape[1]**2)
        qubit_matrix_error[:n] = errors
        np.random.shuffle(qubit_matrix_error)
        matrix[:,:,:] = qubit_matrix_error.reshape(2, matrix.shape[1], matrix.shape[2])

        return matrix

    def isTerminalState(self, state):
        """Evaluates if a state is terminal.

        return - Boolean: True if terminal state, False otherwise
        """
        return np.all(state==0)


    def createSyndrom(self, tcode):
        """Generates the syndrom from the given qubit matrix.
        tcode - the qubit matrix containing the errors.

        return - a matrix with syndroms to corresponding input matrix.
        """
        # generate vertex excitations (charge)
        # can be generated by z and y errors 
        qubit0 = tcode[0,:,:]        
        y_errors = (qubit0 == 2).astype(int) # separate y and z errors from x 
        z_errors = (qubit0 == 3).astype(int)
        charge = y_errors + z_errors # vertex_excitation
        charge_shift = np.roll(charge, 1, axis=0) 
        charge = charge + charge_shift
        charge0 = (charge == 1).astype(int) # annihilate two syndroms at the same place in the grid
        
        qubit1 = tcode[1,:,:]        
        y_errors = (qubit1 == 2).astype(int)
        z_errors = (qubit1 == 3).astype(int)
        charge = y_errors + z_errors
        charge_shift = np.roll(charge, 1, axis=1)
        charge1 = charge + charge_shift
        charge1 = (charge1 == 1).astype(int)
        
        charge = charge0 + charge1
        vertex_matrix = (charge == 1).astype(int)
        
        # generate plaquette excitation (flux)
        # can be generated by x and y errors
        qubit0 = tcode[0,:,:]        
        x_errors = (qubit0 == 1).astype(int)
        y_errors = (qubit0 == 2).astype(int)
        flux = x_errors + y_errors # plaquette_excitation
        flux_shift = np.roll(flux, -1, axis=1)
        flux = flux + flux_shift
        flux0 = (flux == 1).astype(int)
        
        qubit1 = tcode[1,:,:]        
        x_errors = (qubit1 == 1).astype(int)
        y_errors = (qubit1 == 2).astype(int)
        flux = x_errors + y_errors
        flux_shift = np.roll(flux, -1, axis=0)
        flux1 = flux + flux_shift
        flux1 = (flux1 == 1).astype(int)

        flux = flux0 + flux1
        plaquette_matrix = (flux == 1).astype(int)

        return np.stack((vertex_matrix, plaquette_matrix), axis=0)

    def get_reward(self):
        terminal = np.all(self.next_state==0)
        if terminal == True:
            reward = 100
        else:
            defects_state = np.sum(self.state)
            defects_next_state = np.sum(self.next_state)
            reward = defects_state - defects_next_state
        
        return reward

    def syndrom(self, state):
            # generate vertex excitations (charge)
            # can be generated by z and y errors 
            qubit0 = self.qubit_matrix[0,:,:]        
            y_errors = (qubit0 == 2).astype(int) # separate y and z errors from x 
            z_errors = (qubit0 == 3).astype(int)
            charge = y_errors + z_errors # vertex_excitation
            charge_shift = np.roll(charge, 1, axis=0) 
            charge = charge + charge_shift
            charge0 = (charge == 1).astype(int) # annihilate two syndroms at the same place in the grid
            
            qubit1 = self.qubit_matrix[1,:,:]        
            y_errors = (qubit1 == 2).astype(int)
            z_errors = (qubit1 == 3).astype(int)
            charge = y_errors + z_errors
            charge_shift = np.roll(charge, 1, axis=1)
            charge1 = charge + charge_shift
            charge1 = (charge1 == 1).astype(int)
            
            charge = charge0 + charge1
            vertex_matrix = (charge == 1).astype(int)
            
            # generate plaquette excitation (flux)
            # can be generated by x and y errors
            qubit0 = self.qubit_matrix[0,:,:]        
            x_errors = (qubit0 == 1).astype(int)
            y_errors = (qubit0 == 2).astype(int)
            flux = x_errors + y_errors # plaquette_excitation
            flux_shift = np.roll(flux, -1, axis=1)
            flux = flux + flux_shift
            flux0 = (flux == 1).astype(int)
            
            qubit1 = self.qubit_matrix[1,:,:]        
            x_errors = (qubit1 == 1).astype(int)
            y_errors = (qubit1 == 2).astype(int)
            flux = x_errors + y_errors
            flux_shift = np.roll(flux, -1, axis=0)
            flux1 = flux + flux_shift
            flux1 = (flux1 == 1).astype(int)
    
            flux = flux0 + flux1
            plaquette_matrix = (flux == 1).astype(int)
    
            if state == 'state':
                self.current_state = np.stack((vertex_matrix, plaquette_matrix), axis=0)
            elif state == 'next_state':
                self.next_state = np.stack((vertex_matrix, plaquette_matrix), axis=0)
    # Not important now
    def render(self, mode='human'):
        pass

    def close(self):
        pass
