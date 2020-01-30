import gym, gym_ToricCode

env = gym.make('toric-code-v0', size=3, min_qbit_errors=0, p_error=0.1)
state = env.reset()

#size = SYSTEM_SIZE, min_qbit_errors = MIN_QBIT_ERRORS, p_error = P_ERROR