import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env1 import SliceCreationEnv1

env = SliceCreationEnv1()

model = DQN.load("gym-examples/dqn_slices1", env)

obs, info = env.reset()

cont = 0
while cont<100:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print('Action: ', action,'Observation: ', obs, ' | Reward: ', reward, ' | Terminated: ', terminated)
    cont += 1
    if terminated or truncated:
        obs, info = env.reset()