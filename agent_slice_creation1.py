import gymnasium as gym

from stable_baselines3 import DQN

from gym_examples.envs.slice_creation_env1 import SliceCreationEnv1

from os.path import exists

env = SliceCreationEnv1()

#if exists('/data/scripts/DQN_Models/Model 1/dqn_slices1.zip'):
if exists('/home/mario/Documents/DQN_Models/Model 1/gym-examples/dqn_slices1.zip'):
    model = DQN.load("gym-examples/dqn_slices1", env)
else: 
    #model = DQN("MlpPolicy", env, verbose=1, exploration_final_eps=0, exploration_fraction=0.5)
    model = DQN("MlpPolicy", env,
            buffer_size=int(1e5),  # Replay buffer size
            learning_rate=1e-3,     # Learning rate
            learning_starts=80000,  # Number of steps before learning starts
            exploration_fraction=0.5,  # Fraction of total timesteps for exploration
            exploration_final_eps=0,  # Final exploration probability after exploration_fraction * total_timesteps
            train_freq=4,           # Update the model every `train_freq` steps
            gradient_steps=1,       # Number of gradient steps to take after each batch of data
            batch_size=32,          # Number of samples in each batch
            gamma=0.99,             # Discount factor
            tau=1.0,                # Target network update rate
            target_update_interval=1000,  # Interval (in timesteps) at which the target network is updated
            verbose=1)              # Verbosity level

#model = DQN.load("dqn_slices1", env)
#model = DQN("MlpPolicy", env, verbose=1, exploration_final_eps=0, exploration_fraction=0.5)

model.learn(total_timesteps=150000, log_interval=1000)
model.save("gym-examples/dqn_slices1")