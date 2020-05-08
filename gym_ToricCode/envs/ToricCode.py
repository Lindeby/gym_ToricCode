import gym
import numpy as np
import matplotlib.pyplot as plt

# debug
import time, sys

class ToricCode(gym.Env):
    """
    Description:

    Observation:

    Actions:
        the actions avaliable shuld be x, y and z operation on all
        position of the matrix
    Reward:

    Episode Termination:

    """

    metadata = {'render.modes': ['human']}
    
    def __init__(self, config):
        """ Init method

        Parameters
        =============
        config: a dictionary containing configuration parmeters
        config {
            size: An integer describing the size of the toric code.
            min_qbit_errors: An integer describing the minium number of qubit errors on the code. 
            p_error: The probability of generating an error on the toric code. 
        }
        """

        self.system_size = config["size"] if "size" in config else 3
        #self.min_qbit_errors = config["min_qbit_errors"] if "min_qbit_errors" in config else 0
        self.p_error = config["p_error"] if "p_error" in config else 0.1

        low = np.array([0,0,0,0])
        high = np.array([1, self.system_size, self.system_size, 3])
        self.action_space       = gym.spaces.Box(low, high)
        self.observation_space  = gym.spaces.Box(0, 1, [2, self.system_size, self.system_size])

        # plaquette_matrix and vertex_matrix combinded beckomes the state
        # that is avaliable for the agent
        self.plaquette_matrix   = np.zeros((self.system_size, self.system_size), dtype=int)   
        self.vertex_matrix      = np.zeros((self.system_size, self.system_size), dtype=int) 

        # qubit_matrix contains the true errors and shuld not be avaliable for for an agent
        self.qubit_matrix       = np.zeros((2, self.system_size, self.system_size), dtype=int)
        self.state              = np.stack((self.vertex_matrix, self.plaquette_matrix,), axis=0)
        self.next_state         = np.stack((self.vertex_matrix, self.plaquette_matrix), axis=0)

        self.rule_table = np.array(([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]), dtype=int)  # Identity = 0, pauli_x = 1, pauli_y = 2, pauli_z = 3

        self.ground_state = True    # True: only trivial loops, False: non trivial loop 


    def step(self, action):
        """ Applies a pauli operator to the toric grid.

        Parameters
        =============
        action: numpy array [matrix, posx, posy, operator] 
        
        Returns
        =============
        numpy array with corresponing syndrom for the errors.
        """

        qubit_matrix = action[0]
        row = action[1]
        col = action[2]
        add_operator = action[3]

        old_operator = self.qubit_matrix[qubit_matrix, row, col]
        new_operator = self.rule_table[old_operator, add_operator]
        self.qubit_matrix[qubit_matrix, row, col] = new_operator   

        self.next_state = self.createSyndromOpt(self.qubit_matrix)
        
        reward = self.getReward()
        self.state = self.next_state 

        return self.state, reward, self.isTerminalState(self.state), {} 
       
    def reset(self, p_error=None):
        """Resets the environment and generates new errors.

        Returns
        =============
        numpy array with corresponing syndrom for the errors.
        """
        if not p_error is None:
            self.p_error = p_error

        self.ground_state = True

        self.plaquette_matrix   = np.zeros((self.system_size, self.system_size), dtype=int)   # dont use self.plaquette
        self.vertex_matrix      = np.zeros((self.system_size, self.system_size), dtype=int)      # dont use self.vertex 
        self.qubit_matrix       = np.zeros((2, self.system_size, self.system_size), dtype=int)
        self.state              = np.stack((self.vertex_matrix, self.plaquette_matrix,), axis=0)
        self.next_state         = np.stack((self.vertex_matrix, self.plaquette_matrix), axis=0)
        
        # Generate new errors
        while True:
            self.qubit_matrix = self.generateRandomError(self.qubit_matrix, self.p_error)

            self.state = self.createSyndromOpt(self.qubit_matrix) # Create syndrom from errors
            if not self.isTerminalState(self.state):
                break


        # self.qubit_matrix = np.array([[[0,0,0],[0,0,0],[0,0,0]],
        #                               [[0,0,0],[0,1,0],[0,1,0]]
        #                             ])
        #self.qubit_matrix = np.array([  [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
        #                                [[0,0,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]
        #                            ])
        # self.qubit_matrix = np.array([  [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0]],
                    # [[0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]]
                # ])
        # self.state = self.createSyndromOpt(self.qubit_matrix)
        return self.state


    def generateRandomError(self, matrix, p_error):
        """Generates errors with a probability.

        Parameters
        =============
        matrix: (2,n,n) numpy matrix the to generate errors on.
        p_error: the probability to generate an error.
        
        Return
        =============
        The input matrix with newly generated errors.
        """

        qubits = np.random.uniform(0, 1, size=(2, matrix.shape[1], matrix.shape[2]))
        error = qubits > p_error
        no_error = qubits < p_error
        qubits[error] = 0
        qubits[no_error] = 1
        pauli_error = np.random.randint(3, size=(2, matrix.shape[1], matrix.shape[2])) + 1
        matrix = np.multiply(qubits, pauli_error)

        return matrix.astype(np.int)
    

    def createSyndromOpt(self, tcode):
        """Generates the syndrom from the given qubit matrix.
        tcode - the qubit matrix containing the errors.

        return - a matrix with syndroms to corresponding input matrix.
        """
        x_errors = np.where(tcode == 1, 1, 0)
        y_errors = np.where(tcode == 2, 1, 0)   # separate y and z errors from x 
        z_errors = np.where(tcode == 3, 1, 0)

        # generate vertex excitations (charge)
        # can be generated by z and y errors 
        charge = y_errors + z_errors            # vertex_excitation
        charge_shift = np.array([np.roll(charge[0], 1, axis=0), np.roll(charge[1], 1, axis=1)])
        charge = charge + charge_shift
        charge = np.where(charge == 1, 1, 0)    # annihilate two syndroms at the same place in the grid

        charge = charge[0] + charge[1]
        vertex_matrix = np.where(charge == 1, 1, 0)
        

        # generate plaquette excitation (flux)
        # can be generated by x and y errors
        flux = x_errors + y_errors                # plaquette_excitation
        flux_shift = np.array([np.roll(flux[0], -1, axis=1), np.roll(flux[1], -1, axis=0)])
        flux = flux + flux_shift
        flux = np.where(flux == 1, 1, 0)

        flux = flux[0] + flux[1]
        plaquette_matrix = np.where(flux == 1, 1, 0)
        
        return np.stack((vertex_matrix, plaquette_matrix), axis=0)


    def isTerminalState(self, state):
        """Evaluates if a state is terminal.

        return - Boolean: True if terminal state, False otherwise
        """
        return np.all(state==0)

    def getReward(self):
        terminal = np.all(self.next_state==0)
        if terminal == True:
            reward = 100
        else:
            defects_state = np.sum(self.state)
            defects_next_state = np.sum(self.next_state)
            reward = defects_state - defects_next_state
        
        return reward

    def plotToricCode(self, state, title):
        x_error_qubits1 = np.where(self.qubit_matrix[0,:,:] == 1)
        y_error_qubits1 = np.where(self.qubit_matrix[0,:,:] == 2)
        z_error_qubits1 = np.where(self.qubit_matrix[0,:,:] == 3)

        x_error_qubits2 = np.where(self.qubit_matrix[1,:,:] == 1)
        y_error_qubits2 = np.where(self.qubit_matrix[1,:,:] == 2)
        z_error_qubits2 = np.where(self.qubit_matrix[1,:,:] == 3)

        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        vertex_defect_coordinates = np.where(vertex_matrix)
        plaquette_defect_coordinates = np.where(plaquette_matrix)

        #xLine = np.linspace(0, self.system_size-0.5, self.system_size)
        xLine = np.linspace(0, self.system_size, self.system_size)
        x = range(self.system_size)
        X, Y = np.meshgrid(x,x)
        XLine, YLine = np.meshgrid(x, xLine)

        markersize_qubit = 15
        markersize_excitation = 7
        markersize_symbols = 7
        linewidth = 2

        ax = plt.subplot(111)
        ax.plot(XLine, -YLine, 'black', linewidth=linewidth)
        ax.plot(YLine, -XLine, 'black', linewidth=linewidth)
        
        # add the last two black lines 
        ax.plot(XLine[:,-1] + 1.0, -YLine[:,-1], 'black', linewidth=linewidth)
        ax.plot(YLine[:,-1], -YLine[-1,:], 'black', linewidth=linewidth)

        ax.plot(X + 0.5, -Y, 'o', color = 'black', markerfacecolor = 'white', markersize=markersize_qubit+1)
        ax.plot(X, -Y -0.5, 'o', color = 'black', markerfacecolor = 'white', markersize=markersize_qubit+1)
        # add grey qubits
        ax.plot(X[-1,:] + 0.5, -Y[-1,:] - 1.0, 'o', color = 'black', markerfacecolor = 'grey', markersize=markersize_qubit+1)
        ax.plot(X[:,-1] + 1.0, -Y[:,-1] - 0.5, 'o', color = 'black', markerfacecolor = 'grey', markersize=markersize_qubit+1)
        
        # all x errors 
        ax.plot(x_error_qubits1[1], -x_error_qubits1[0] - 0.5, 'o', color = 'r', label="x error", markersize=markersize_qubit)
        ax.plot(x_error_qubits2[1] + 0.5, -x_error_qubits2[0], 'o', color = 'r', markersize=markersize_qubit)
        ax.plot(x_error_qubits1[1], -x_error_qubits1[0] - 0.5, 'o', color = 'black', markersize=markersize_symbols, marker=r'$X$')    
        ax.plot(x_error_qubits2[1] + 0.5, -x_error_qubits2[0], 'o', color = 'black', markersize=markersize_symbols, marker=r'$X$')

        # all y errors
        ax.plot(y_error_qubits1[1], -y_error_qubits1[0] - 0.5, 'o', color = 'blueviolet', label="y error", markersize=markersize_qubit)
        ax.plot(y_error_qubits2[1] + 0.5, -y_error_qubits2[0], 'o', color = 'blueviolet', markersize=markersize_qubit)
        ax.plot(y_error_qubits1[1], -y_error_qubits1[0] - 0.5, 'o', color = 'black', markersize=markersize_symbols, marker=r'$Y$')
        ax.plot(y_error_qubits2[1] + 0.5, -y_error_qubits2[0], 'o', color = 'black', markersize=markersize_symbols, marker=r'$Y$')

        # all z errors 
        ax.plot(z_error_qubits1[1], -z_error_qubits1[0] - 0.5, 'o', color = 'b', label="z error", markersize=markersize_qubit)
        ax.plot(z_error_qubits2[1] + 0.5, -z_error_qubits2[0], 'o', color = 'b', markersize=markersize_qubit)
        ax.plot(z_error_qubits1[1], -z_error_qubits1[0] - 0.5, 'o', color = 'black', markersize=markersize_symbols, marker=r'$Z$')
        ax.plot(z_error_qubits2[1] + 0.5, -z_error_qubits2[0], 'o', color = 'black', markersize=markersize_symbols  , marker=r'$Z$')


        #ax.plot(vertex_defect_coordinates[1], -vertex_defect_coordinates[0], 'x', color = 'blue', label="charge", markersize=markersize_excitation)
        ax.plot(vertex_defect_coordinates[1], -vertex_defect_coordinates[0], 'o', color = 'blue', label="charge", markersize=markersize_excitation)
        ax.plot(plaquette_defect_coordinates[1] + 0.5, -plaquette_defect_coordinates[0] - 0.5, 'o', color = 'red', label="flux", markersize=markersize_excitation)
        ax.axis('off')
        
        #plt.title(title)
        plt.axis('equal')
        plt.savefig('plots/graph_'+str(title)+'.png')
        plt.close()


    def evalGroundState(self):
        """ Evaluates ground state of the toric code. Can only
        distinguish non trivial and trivial loop. Categorization
        what kind of non trivial loop does not work.
        
        Note: Function works only for odd grid dimensions! 3x3, 5x5, 7x7   

        Return
        ======
        (Bool) True if there are trivial loops, False for non-trivial loops.
        """
        def split_qubit_matrix_in_x_and_z():
        # loops vertex space qubit matrix 0
            z_matrix_0 = self.qubit_matrix[0,:,:]        
            y_errors = (z_matrix_0 == 2).astype(int)
            z_errors = (z_matrix_0 == 3).astype(int)
            z_matrix_0 = y_errors + z_errors 
            # loops vertex space qubit matrix 1
            z_matrix_1 = self.qubit_matrix[1,:,:]        
            y_errors = (z_matrix_1 == 2).astype(int)
            z_errors = (z_matrix_1 == 3).astype(int)
            z_matrix_1 = y_errors + z_errors
            # loops plaquette space qubit matrix 0
            x_matrix_0 = self.qubit_matrix[0,:,:]        
            x_errors = (x_matrix_0 == 1).astype(int)
            y_errors = (x_matrix_0 == 2).astype(int)
            x_matrix_0 = x_errors + y_errors 
            # loops plaquette space qubit matrix 1
            x_matrix_1 = self.qubit_matrix[1,:,:]        
            x_errors = (x_matrix_1 == 1).astype(int)
            y_errors = (x_matrix_1 == 2).astype(int)
            x_matrix_1 = x_errors + y_errors

            return x_matrix_0, x_matrix_1, z_matrix_0, z_matrix_1

        x_matrix_0, x_matrix_1, z_matrix_0, z_matrix_1 = split_qubit_matrix_in_x_and_z()
        
        loops_0 = np.sum(np.sum(x_matrix_0, axis=0))
        loops_1 = np.sum(np.sum(x_matrix_1, axis=0))
        
        loops_2 = np.sum(np.sum(z_matrix_0, axis=0))
        loops_3 = np.sum(np.sum(z_matrix_1, axis=0))

        if loops_0%2 == 1 or loops_1%2 == 1:
            self.ground_state = False
        elif loops_2%2 == 1 or loops_3%2 == 1:
            self.ground_state = False

        return self.ground_state


    def render(self, mode='human'):
        pass

    def close(self):
        pass