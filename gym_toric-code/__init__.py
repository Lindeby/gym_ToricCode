from gym.envs.registration import register

register(
    id='toric-code-v0',
    entry_point='gym_toric-code.envs:Toric-codeEnv',
)
