import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env1 import SliceCreationEnv1

from os.path import exists

env = SliceCreationEnv1()


if exists('/home/mario/Documents/DQN_Models/Model 1/gym-examples/dqn_slices1.zip'):
    model = DQN.load("dqn_slices1", env)
else: model = DQN("MlpPolicy", env, verbose=1, exploration_final_eps=0, exploration_fraction=0.5)

#model = DQN.load("dqn_slices1", env)
#model = DQN("MlpPolicy", env, verbose=1, exploration_final_eps=0, exploration_fraction=0.5)

model.learn(total_timesteps=5000, log_interval=1000)
model.save("dqn_slices1")