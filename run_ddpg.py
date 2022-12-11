from __future__ import print_function
from collections import deque

from rl.pg_ddpg import DeepDeterministicPolicyGradient
import tensorflow as tf
import numpy as np
import sys
# sys.path.append('/home/stone/gym/')
import gym

# env_name = 'Hopper-v1'
# env_name = 'Swimmer-v1'
# env_name = 'HalfCheetah-v1'
env_name = 'Walker2d-v1'
# env_name = 'InvertedPendulum-v1'
# env_name = 'InvertedDoublePendulum-v1'
# env_name = 'Reacher-v1'

env = gym.make(env_name)
sess      = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
# writer    = tf.train.SummaryWriter("/tmp/{}-experiment-1".format(env_name))

state_dim  = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# DDPG actor and critic architecture
# Continuous control with deep reinforcement learning
# Timothy P. Lillicrap, et al., 2015

def actor_network(states):
  h1_dim = 400
  h2_dim = 300

  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, h1_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable("b1", [h1_dim],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(states, W1) + b1)

  W2 = tf.get_variable("W2", [h1_dim, h2_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable("b2", [h2_dim],
                       initializer=tf.constant_initializer(0))
  h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

  # use tanh to bound the action
  W3 = tf.get_variable("W3", [h2_dim, action_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b3 = tf.get_variable("b3", [action_dim],
                       initializer=tf.constant_initializer(0))

  # we assume actions range from [-1, 1]
  # you can scale action outputs with any constant here
  a = tf.nn.tanh(tf.matmul(h2, W3) + b3)
  return a

def critic_network(states, action):
  h1_dim = 400
  h2_dim = 300

  # define policy neural network
  W1 = tf.get_variable("W1", [state_dim, h1_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b1 = tf.get_variable("b1", [h1_dim],
                       initializer=tf.constant_initializer(0))
  h1 = tf.nn.relu(tf.matmul(states, W1) + b1)
  # skip action from the first layer
  h1_concat = tf.concat([h1, action], 1)

  W2 = tf.get_variable("W2", [h1_dim + action_dim, h2_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
  b2 = tf.get_variable("b2", [h2_dim],
                       initializer=tf.constant_initializer(0))
  h2 = tf.nn.relu(tf.matmul(h1_concat, W2) + b2)

  W3 = tf.get_variable("W3", [h2_dim, 1],
                       initializer=tf.contrib.layers.xavier_initializer())
  b3 = tf.get_variable("b3", [1],
                       initializer=tf.constant_initializer(0))
  v = tf.matmul(h2, W3) + b3
  return v

pg_ddpg = DeepDeterministicPolicyGradient(sess,
                                          optimizer,
                                          actor_network,
                                          critic_network,
                                          state_dim,
                                          action_dim) # ,
                                          # summary_writer=writer)

MAX_EPISODES = 5000 # 100000
#MAX_STEPS = 50 # if testing "Reacher" task
MAX_STEPS    = 5000 # 1000
mean_rewards = 0

record = [0 for x in range(0, 10000)]


f=open('/home/cardwing/Desktop/usefulcode_HouYuenan/reward_curve/PER-DDPG/walker2d.txt','w')
episode_history = deque(maxlen=100)#record total reward
# reward_history  = deque(maxlen=1000)
for i_episode in xrange(MAX_EPISODES):
  #if i_episode == 300:
  #  MAX_STEPS  = 5000
  # initialize
  state = env.reset()
  total_rewards = 0
  # print(MAX_STEPS)
  # run for an episode
  for t in xrange(MAX_STEPS):
    # env.render()
    action = pg_ddpg.sampleAction(state[np.newaxis,:])
    next_state, reward, done, _ = env.step(action)
    pg_ddpg.storeExperience(state, action, reward, next_state, done)
    error = pg_ddpg.updateModel(record)
    total_rewards += reward
    state = next_state
    if done: break
  episode_history.append(total_rewards)
  f.write(str(total_rewards))
  f.write('\n')
  if i_episode % 100 == 0:
    for i_eval in range(100):
      total_rewards = 0
      state = env.reset()
      for t in xrange(MAX_STEPS):
        if i_episode >= 400 and mean_rewards > 8000:
          env.render()
        action = pg_ddpg.sampleAction(state[np.newaxis,:], exploration=False)
        next_state, reward, done, _ = env.step(action)
        total_rewards += reward
        state = next_state
        if done: break
      episode_history.append(total_rewards)
    mean_rewards = np.mean(episode_history)

    print("Episode {}".format(i_episode))
    print("Finished after {} timesteps".format(t+1))
    print("Reward for this episode: {:.2f}".format(total_rewards))
    print("Average reward for last 100 episodes: {:.2f}".format(mean_rewards))
    #if mean_rewards >= -3.75:# for Reacher-v1
    # if mean_rewards >= 950.0: # for InvertedPendulum-v1
    if mean_rewards >= 9100.0 or i_episode >= 5000: # for InvertedDoublePendulum-v1
    #if mean_rewards >= -3: # for Walker2d-v1
      print("Environment {} solved after {} episodes".format(env_name, i_episode+1))
      f.close()
      break
f.close()