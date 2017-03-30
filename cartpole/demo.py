# Cartpole:
# Action_space = Discrete(2) - 1 moves to right, 0 moves to left
import gym
env = gym.make('CartPole-v0')
for episode in range(20):
    obs = env.reset()
    for t in range(100):
        env.render()
        print(obs)
        action = env.action_space.sample()
        obs, reward, done, info = env.step(1)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break;
