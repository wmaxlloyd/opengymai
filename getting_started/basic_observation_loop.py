import gym
env = gym.make('FrozenLake-v0')
for i_episode in range(2):
    observation = env.reset()
    print(env.action_space)
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print({
            "obesrvation": observation,
            "reward": reward,
            "done": done,
            "info": info
        })
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()