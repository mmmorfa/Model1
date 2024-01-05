import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env import SliceCreationEnv

env = SliceCreationEnv()

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)
model.save("dqn_slices")


'''
del model # remove to demonstrate saving and loading

model = DQN.load("dqn_slices")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
'''