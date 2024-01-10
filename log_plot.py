import csv
import matplotlib.pyplot as plt

def read_csv(file_path):
    columns = {}  # Dictionary to store lists for each column
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        
        headers = next(reader, None)
        
        if headers:
            # Initialize lists for each column
            for header in headers:
                columns[header] = []
            
            # Read and store data in respective columns
            for row in reader:
                for header, value in zip(headers, row):
                    columns[header].append(value)
    
    return columns

file_path1 = '/home/mario/Documents/DQN_Models/Model 1/gym-examples/logs/progress(1e-4).csv'
#file_path = '/data/scripts/DQN_models/Model 1/logs/progress.csv'

file_path2 = '/home/mario/Documents/DQN_Models/Model 1/gym-examples/logs/progress(1e-3).csv'
#file_path3 = '/home/mario/Documents/DQN_Models/Model 1/gym-examples/logs/progress(1e-5).csv'

data1 = read_csv(file_path1)
ep_rew_mean1 = [float(i) for i in data1['rollout/ep_rew_mean']]
ep_len_mean1 = [float(i) for i in data1['rollout/ep_len_mean']]
exploration_rate1 = [float(i) for i in data1['rollout/exploration_rate']]
episodes1 = [float(i) for i in data1['time/episodes']]
fps1 = [float(i) for i in data1['time/fps']]
time_elapsed1 = [float(i) for i in data1['time/time_elapsed']]
total_timesteps1 = [float(i) for i in data1['time/total_timesteps']]
learning_rate1 = [float(i) for i in data1['train/learning_rate'] if i!= '']
loss1 = [float(i) for i in data1['train/loss'] if i!= '']
n_updates1 = [float(i) for i in data1['train/n_updates'] if i!= '']


data2 = read_csv(file_path2)
ep_rew_mean2 = [float(i) for i in data2['rollout/ep_rew_mean']]
ep_len_mean2 = [float(i) for i in data2['rollout/ep_len_mean']]
exploration_rate2 = [float(i) for i in data2['rollout/exploration_rate']]
episodes2 = [float(i) for i in data2['time/episodes']]
fps2 = [float(i) for i in data2['time/fps']]
time_elapsed2 = [float(i) for i in data2['time/time_elapsed']]
total_timesteps2 = [float(i) for i in data2['time/total_timesteps']]
learning_rate2 = [float(i) for i in data2['train/learning_rate'] if i!= '']
loss2 = [float(i) for i in data2['train/loss'] if i!= '']
n_updates2 = [float(i) for i in data2['train/n_updates'] if i!= '']

'''
data3 = read_csv(file_path3)
ep_rew_mean3 = [float(i) for i in data3['rollout/ep_rew_mean']]
ep_len_mean3 = [float(i) for i in data3['rollout/ep_len_mean']]
exploration_rate3 = [float(i) for i in data3['rollout/exploration_rate']]
episodes3 = [float(i) for i in data3['time/episodes']]
fps3 = [float(i) for i in data3['time/fps']]
time_elapsed3 = [float(i) for i in data3['time/time_elapsed']]
total_timesteps3 = [float(i) for i in data3['time/total_timesteps']]
learning_rate3 = [float(i) for i in data3['train/learning_rate'] if i!= '']
loss3 = [float(i) for i in data3['train/loss'] if i!= '']
n_updates3 = [float(i) for i in data3['train/n_updates'] if i!= '']
'''

plt.plot(total_timesteps1, ep_rew_mean1, marker='o', linestyle='-', color='b', label='Learning rate = 1e-4')
plt.plot(total_timesteps2, ep_rew_mean2, marker='o', linestyle='-', color='r', label='Learning rate = 1e-3')
#plt.plot(total_timesteps3, ep_rew_mean3, marker='o', linestyle='-', color='g', label='Learning rate = 1e-5')
plt.xlabel("Timesteps")
plt.ylabel("Episode Mean Reward")
plt.title("Training plots")
plt.legend()
plt.show()

