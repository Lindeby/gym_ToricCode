import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

# debug
import time, sys

class ToricCodeCUDA(gym.Env):
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
            size:               (Int)   Size of the toric code.
            min_qbit_errors:    (Int)   Minium number of qubit errors on the code. 
            p_error:            (Float) The probability of generating an error on the toric code. 
            device:             (String) {"cuda", "cpu"} To run the tensor operations on.
        }
        """

        self.system_size = config["size"] if "size" in config else 3
        self.min_qbit_errors = config["min_qbit_errors"] if "min_qbit_errors" in config else 0
        self.p_error = config["p_error"] if "p_error" in config else 0.1
        self.device = torch.device(config["device"]) if "device" in config else torch.device('cpu')

        low = np.array([0,0,0,0])
        high = np.array([1, self.system_size, self.system_size, 3])
        self.action_space       = gym.spaces.Box(low, high)
        self.observation_space  = gym.spaces.Box(0, 1, [2, self.system_size, self.system_size])

        self.plaquette_matrix   = torch.zeros((self.system_size, self.system_size), dtype=torch.int8, device=self.device)
        self.vertex_matrix      = torch.zeros((self.system_size, self.system_size), dtype=torch.int8, device=self.device)

        # qubit_matrix contains the true errors and shuld not be avaliable for for an agent    
        self.qubit_matrix       = torch.zeros((self.system_size, self.system_size), dtype=torch.int8, device=self.device)
        self.state              = torch.stack((self.vertex_matrix, self.plaquette_matrix), dim=0)
        self.next_state         = torch.stack((self.vertex_matrix, self.plaquette_matrix), dim=0)

        self.rule_table = torch.Tensor([[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]], device=self.device)  # Identity = 0, pauli_x = 1, pauli_y = 2, pauli_z = 3

        self.ground_state = True    # True: only trivial loops, False: non trivial loop

        # utility
        self.one  = torch.ones( (2,self.system_size, self.system_size), device=self.device)
        self.zero = torch.zeros((2,self.system_size, self.system_size), device=self.device)


    def step(self, action):
        """ Applies a pauli operator to the toric grid.

        Params
        =============
        action: (torch.Tensor) [matrix, posx, posy, operator] 
        
        Returns
        =============
        (torch.Tensor)
        (torch.Tensor)
        (torch.Tensor)
        (dict)
        """

        matrix, row, col, op = action.type(torch.long)

        old_operator = self.qubit_matrix[matrix][row][col].type(torch.long)
        new_operator = self.rule_table[old_operator][op]
        self.qubit_matrix[matrix][row][col] = new_operator

        self.next_state = self.createSyndromOpt(self.qubit_matrix)
        
        terminal = self.isTerminalState(self.next_state)
        reward = torch.Tensor([100], device=self.device) if terminal else self.state.sum()-self.next_state.sum()
        self.state = self.next_state 

        return self.state, reward, terminal, {} 
    

    def reset(self):
        """Resets the environment and generates new errors.

        Returns
        =============
        (torch.Tensor)
        """

        self.ground_state = True

        self.plaquette_matrix   = torch.zeros((self.system_size, self.system_size), dtype=torch.int8, device=self.device)
        self.vertex_matrix      = torch.zeros((self.system_size, self.system_size), dtype=torch.int8, device=self.device)
        # self.state              = torch.stack((self.vertex_matrix, self.plaquette_matrix), dim=0)
        self.next_state         = torch.stack((self.vertex_matrix, self.plaquette_matrix), dim=0)
        

        # TODO: Cant guarante that the new state will contain errors
        self.qubit_matrix = self.generateRandomError(self.p_error)
        terminal = self.isTerminalState(self.state)

        # Debugging
        # self.qubit_matrix = torch.Tensor([  [[0,0,0],[0,0,0],[0,0,0]],
        #                                [[0,0,0],[0,1,0],[0,1,0]]
        #                            ])
        # self.qubit_matrix = torch.Tensor([  [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]],
        #                                [[0,0,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]]
        #                            ])
        

        self.state = self.createSyndromOpt(self.qubit_matrix) # Create syndrom from errors
        return self.state


    def generateRandomError(self, p_error):
        """Generates errors with a probability.

        Params
        ======
        p_error: (float)
        
        Return
        =============
        (torch.Tensor)
        """
        
        qubits = torch.zeros((2, self.system_size, self.system_size), device=self.device).uniform_(0,1)
        error  = torch.where(qubits < p_error, self.one, self.zero)
        pauli_error  = torch.zeros((2, self.system_size, self.system_size), device=self.device).random_(1,4)
        return error * pauli_error

    
    def generateNRandomErrors(self, matrix, n):
        """Generates n errors.
        np matrix - a (2,x,x) numpy matrix the to generate errors on.
        int n - the number of generated errors.

        return - the error matrix
        """ 
        pass
        # errors = np.random.randint(3, size = n) + 1
        # qubit_matrix_error = np.zeros(2*matrix.shape[1]**2)
        # qubit_matrix_error[:n] = errors
        # np.random.shuffle(qubit_matrix_error)
        # matrix[:,:,:] = qubit_matrix_error.reshape(2, matrix.shape[1], matrix.shape[2])

        # return matrix

    def createSyndromOpt(self, tcode):
        """Generates the syndrom from the given qubit matrix.

        Params
        ======
        tcode:  (torch.Tensor)

        Return
        ======
        (torch.Tensor)
        """
        one  = self.one
        zero = self.zero

        x_errors = torch.where(tcode == 1, one, zero)
        y_errors = torch.where(tcode == 2, one, zero) 
        z_errors = torch.where(tcode == 3, one, zero)

        # generate vertex excitations (charge)
        # can be generated by z and y errors 
        charge = y_errors + z_errors            # vertex_excitation
        charge_shift = torch.stack((torch.roll(charge[0], 1, dims=0), torch.roll(charge[1], 1, dims=1)))
        charge = charge + charge_shift
        charge = torch.where(charge == 1, one, zero)    # annihilate two syndroms at the same place in the grid

        charge = charge[0] + charge[1]
        vertex_matrix = torch.where(charge == 1, one[0], zero[0])
        
        # generate plaquette excitation (flux)
        # can be generated by x and y errors
        flux = x_errors + y_errors                # plaquette_excitation
        flux_shift = torch.stack((torch.roll(flux[0], -1, dims=1), torch.roll(flux[1], -1, dims=0)))
        flux = flux + flux_shift
        flux = torch.where(flux == 1, one, zero)

        flux = flux[0] + flux[1]
        plaquette_matrix = torch.where(flux == 1, one[0], zero[0])
        
        return torch.stack((vertex_matrix, plaquette_matrix), dim=0)


    def isTerminalState(self, state):
        """Evaluates if a state is terminal.

        return - Boolean: True if terminal state, False otherwise
        """
        return state.sum() == 0



    def plotToricCode(self, state, title):
        st = torch.from_numpy(state)
        qm = torch.from_numpy(self.qubit_matrix)
        x_error_qubits1 = np.where(qm[0,:,:] == 1)
        y_error_qubits1 = np.where(qm[0,:,:] == 2)
        z_error_qubits1 = np.where(qm[0,:,:] == 3)
        x_error_qubits2 = np.where(qm[1,:,:] == 1)
        y_error_qubits2 = np.where(qm[1,:,:] == 2)
        z_error_qubits2 = np.where(qm[1,:,:] == 3)

        vertex_matrix = st[0,:,:]
        plaquette_matrix = st[1,:,:]
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

        one = self.one
        zero = self.zero

        # loops vertex space qubit matrix 0
        # y error + z error
        z_matrix_0 = self.qubit_matrix[0]
        z_matrix_0 = torch.where(z_matrix_0==2, one, zero) + torch.where(z_matrix_0==3, one, zero)

        # loops vertex space qubit matrix 1
        # y error + z error
        z_matrix_1 = self.qubit_matrix[1]
        z_matrix_1 = torch.where(z_matrix_1==2, one, zero) + torch.where(z_matrix_1==3, one, zero)

        # loops plaquette space qubit matrix 0
        # x error + y error
        x_matrix_0 = self.qubit_matrix[0]
        x_matrix_0 = torch.where(x_matrix_0==1, one, zero) + torch.where(x_matrix_0==2, one, zero)

        # loops plaquette space qubit matrix 1
        # x error + y error
        x_matrix_1 = self.qubit_matrix[1]
        x_matrix_1 = torch.where(x_matrix_1==1, one, zero) + torch.where(x_matrix_1==2, one, zero)

        loops_0 = x_matrix_0.sum()
        loops_1 = x_matrix_1.sum()
        
        loops_2 = z_matrix_0.sum()
        loops_3 = z_matrix_1.sum()

        if loops_0%2 == 1 or loops_1%2 == 1:
            self.ground_state = False
        elif loops_2%2 == 1 or loops_3%2 == 1:
            self.ground_state = False

        return self.ground_state


    def render(self, mode='human'):
        pass

    def close(self):
        pass
