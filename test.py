import gym, gym_ToricCode, torch, time
import numpy as np


size = 3
device = 'cpu'
p_error = 0.5
one  = torch.ones( (2,size, size), device=device)
zero = torch.zeros((2,size, size), device=device)


def error(size, device, p_error):
    qubits = torch.zeros((2, size, size), device=device).uniform_(0,1)
    error  = torch.where(qubits < p_error, one, zero)
    pauli_error  = torch.zeros((2, size, size), device=device).random_(1,4)


if __name__ == "__main__":
    # conf = {
    #         "size":size
    #         , "min_qbit_errors":0 
    #         , "p_error":0.1 
    #         , "device": device
    #     }

    # env = gym.make('toric-code-v0', config=conf)
    # env_cuda = gym.make('toric-code-cuda-v0', config=conf)
    itr = 1000000
    start = time.time()
    for i in range(itr):
        error(size, device, p_error)
    end = time.time()

    print("Total elapsed time: {}".format(end-start))


