import gym
import pybullet, pybullet_envs
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('BipedalWalker-v3')
env.render(mode='human')
policy_kwargs = dict(activation_fn=th.nn.LeakyReLU, net_arch=[512, 512])

# Instantiate the agent
model = PPO('MlpPolicy', env, learning_rate = 0.0003, policy_kwargs = policy_kwargs, verbose = 1)

del model
model = PPO.load("PPO_BipedalWalker-v3_0.zip", env = env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward}")

obs = env.reset()
for i in range(100):
    dones = False
    game_score = 0
    steps = 0
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        game_score += rewards
        steps += 1
        env.render()
    print