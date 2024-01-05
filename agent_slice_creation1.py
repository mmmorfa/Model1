import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env1 import SliceCreationEnv1

env = SliceCreationEnv1()

model = DQN("MlpPolicy", env, verbose=1, exploration_final_eps=0, exploration_fraction=0.5)
model.learn(total_timesteps=100, log_interval=1000)
model.save("dqn_slices1")


'''
del model # remove to demonstrate saving and loading

model = DQN.load("dqn_slices1")

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
'''