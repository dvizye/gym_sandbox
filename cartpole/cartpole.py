# Cartpole Random linear controller
# Just samples params to dot product with the state to generate actions
#
# Cartpole env:
# Action_space = Discrete(2) - 1 moves to right, 0 moves to left
# Obs space: Box(-4.8 to 4.8, -inf to inf, -0.42 to 0.42, -inf to inf)
# I assume this corresponds to [theta, pos, theta_dot, pos_dot]
# done when abs(theta) > 0.15

import gym
import numpy as np

def run_episode(env, controller, duration, render=False):
    """Run one gym episode

    Args:
        env: env to run
        controller: function that takes in obs and return action
        duration: time_duration to run episode if done
        render: whether to render the episode
    Returns:
        Int cumulative reward
    """
    cum_reward = 0
    obs = env.reset()
    for t in xrange(duration):
        if render:
            env.render()
        action = controller(obs)
        obs, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break
    return cum_reward


def sample_rand_linear_controllers(env, num_samples):
    num_params = env.observation_space.shape[0]
    rand_params = np.random.random([num_samples, 1, num_params])* 2 - 1
    rewards = np.zeros(num_samples)
    for i in range(num_samples):
        params = rand_params[i]
        reward = run_episode(env, lambda obs: linear_control(obs, params), 10000)
        rewards[i] = reward
    return rewards, rand_params

def linear_control(obs, params):
    return int(np.dot(params, obs) > 0)

env = gym.make('CartPole-v0')
rewards, params = sample_rand_linear_controllers(env, 50)
print rewards
i = np.argmax(rewards)
best_params = params[i]
print 'best params index'
print i
obs = env.reset()
cum_reward = 0
for t in range(1000):
    env.render()
    obs, rew, done, _ = env.step(linear_control(obs, best_params))
    cum_reward += rew
    if done:
        break
print 'Final cumulative reward: {}'.format(cum_reward)
