from gym.envs.registration import register

register(
    id='toric-code-v0',
    entry_point='gym_ToricCode.envs:ToricCode',
)

# register(
#     id='toric-code-cuda-v0',
#     entry_point='gym_ToricCode.envs:ToricCodeCUDA',
# )
