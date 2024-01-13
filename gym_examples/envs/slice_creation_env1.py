import gymnasium as gym
#from gym import spaces
import pygame
import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env
from copy import deepcopy
#import os


class SliceCreationEnv1(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        # Define environment parameters
        
        #Available resources (Order: MEC BW, )
        self.resources = [100]
        
        #Defined parameters per Slice. (Each component is a list of the correspondent slice parameters)
        self.slices_param = [10, 20, 50]

        self.slice_requests = pd.read_csv('/home/mario/Documents/DQN_Models/Model 1/gym-examples/gym_examples/slice_request_db1')  # Load VNF requests from the generated CSV
        #self.slice_requests = pd.read_csv('/data/scripts/DQN_models/Model1/gym_examples/slice_request_db1')    #For pod
        
        self.observation_space = gym.spaces.Box(low=0, high=10000, shape=(2,), dtype=np.float32) #ovservation space composed by Requested resources (MEC BW) and available MEC resources.
        
        self.action_space = gym.spaces.Discrete(4)  # 0: Do Nothing, 1: Allocate Slice 1, 2: Allocate Slice 2, 3: Allocate Slice 3

        #self.process_requests()
        
        # Define other necessary variables and data structures
        self.current_time_step = 1
        self.reward = 0
        self.first = True
        
        self.processed_requests = []

    def reset(self, seed=None, options=None):
        # Initialize the environment to its initial state

        #print(self.processed_requests)

        #os.system("gym-examples/gym_examples/vnf_generator.py")

        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.current_time_step = 1
        self.reward = 0
        self.processed_requests = []
        self.reset_resources()
        self.next_request = self.read_request()
        self.update_slice_requests(self.next_request)
        self.observation = np.array([self.next_request[1]] + deepcopy(self.resources), dtype=np.float32)
        #self.observation = np.array(self.observation, np.float32)
        self.info = {}
        self.first = True
        
        #print("\nReset: ", self.observation)
        
        return self.observation, self.info



    def step(self, action):
        
        if self.first:
            self.next_request = self.processed_requests[0]
            self.first = False
        else: 
            #next_request = self.read_request()
            self.update_slice_requests(self.next_request)
            
        terminated = False
        
        slice_id = self.create_slice(self.next_request)
        
        reward_value = 10
        
        # Apply the selected action (0: Do Nothing, 1: Allocate Slice 1, 2: Allocate Slice 2, 3: Allocate Slice 3)
        
        if action == 1 and slice_id == 1:
            if self.check_resources(self.next_request[1]):
                self.allocate_slice(self.next_request[1])
                self.processed_requests[len(self.processed_requests) - 1].append(slice_id)
                self.reward += reward_value   
                self.next_request = self.read_request()
            else: terminated = True
        
        if action == 1 and slice_id != 1:
            terminated = True
            
        if action == 2 and slice_id == 2:
            if self.check_resources(self.next_request[1]):
                self.allocate_slice(self.next_request[1])
                self.processed_requests[len(self.processed_requests) - 1].append(slice_id)
                self.reward += reward_value 
                self.next_request = self.read_request()
            else: terminated = True
        
        if action == 2 and slice_id != 2:
            terminated = True
            
        if action == 3 and slice_id == 3:
            if self.check_resources(self.next_request[1]):
                self.allocate_slice(self.next_request[1])
                self.processed_requests[len(self.processed_requests) - 1].append(slice_id)
                self.reward += reward_value     
                self.next_request = self.read_request()
            else: terminated = True
        
        if action == 3 and slice_id != 3:
            terminated = True
            
        if action == 0:
            if not self.check_resources(self.next_request[1]):
                self.reward += reward_value
                self.next_request = self.read_request()
            else: terminated = True        
    
        self.observation = np.array([self.next_request[1]] + self.resources, dtype=np.float32)
        
        reward = self.reward
        
        #done = False
        
        info = {}  # Additional information (if needed)
        
        #self.current_time_step += 1  # Increment the time step
        
        #print("Action: ", action, "\nObservation: ", self.observation, "\nReward: ", self.reward)
        
        return self.observation, reward, terminated, False, info
    
    def read_request(self):
        next_request = self.slice_requests.iloc[self.current_time_step - 1]
        request_list =list([next_request['ARRIVAL_REQUEST_@TIME'], next_request['SLICE_BW_REQUEST'], next_request['SLICE_KILL_@TIME']])
        self.current_time_step += 1
        return request_list
        

    def update_slice_requests(self, request):
        # Update the slice request list by deleting the killed VNFs
        if len(self.processed_requests) != 0:
            for i in self.processed_requests:
                #i[2] < request[0]
                if i[2] < request[0]:
                    self.deallocate_slice(i)
        self.processed_requests.append(request)
        

    def check_resources(self, slice_bw_request):
        # Logic to check if there are available resources to allocate the VNF request
        # Return True if resources are available, False otherwise
        if self.resources[0] >= int(slice_bw_request):
            return True
        else: return False
    
    def allocate_slice(self, slice_bw_request):
        # Allocate the resources requested by the current VNF
        self.resources[0] -= int(slice_bw_request)
        # Define Slice ID
    
    def deallocate_slice(self, request):
        # Function to deallocate resources of killed requests
        self.resources[0] = self.resources[0] + request[1]
        
    def create_slice (self, request):
        # Function to create the slice for a specific request
        # This function inserts the defined slice to the request in the processed requests list
        
        resources = request[1]
        if resources <= self.slices_param[0]:
            slice_id = 1
        elif resources <= self.slices_param[1]:
            slice_id = 2
        elif resources <= self.slices_param[2]:
            slice_id = 3
        return slice_id

    def reset_resources(self):
        self.resources = [500]
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            
#a = SliceCreationEnv1()
#check_env(a)