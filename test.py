import gym, gym_ToricCode
env = gym.make('toric-code-v0', config={})
state = env.reset()
env.plotToricCode(state, "name")
state, _, _, _ = env.step(env.Action([1,0,1],2))