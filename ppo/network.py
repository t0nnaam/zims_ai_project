
import torch
import torch.nn as nn
import numpy as np

#this is the number of control outputs your agent 
#(the agent is the whole decision making system ie the whole bot)
#produces at each time step. 
#in this case, the output is the 
#input for our 12 servo motors (3 on each leg, to move the joints)
ACTION_SPACE_SIZE = 12

#Actor state input consists of IMU (6), Servo (12) if Lidar is added (3)

#This file defines 2 neural networks for PPO: 
# -actor: decides what action to take. is the bot basically
# -critic: evaluates how good the state is 

class Actor(nn.Module): 
    def __init__(self):
        super(Actor, self).__init__()
        
        #this is for the IMUs (Inertial measurement units)
        #takes 6 IMU numbers and turns them into 64 neurons, which have weights inside
        self.imu_fc1 = nn.Linear(6, 64)
        #takes the 64 numbers and refines the neurons to be better
        self.imu_fc2 = nn.Linear(64, 32)
        
        #a neuron multiples inputs by weights, adds toegther and passes them through the 
        #activation function 
        
        #for the servos
        self.servo_fc1 = nn.Linear(12, 64)
        self.servo_fc2 = nn.Linear(64, 32)
        
        #for the lidar 
        self.lidar_fc1 = nn.Linear(3, 64)
        self.lidar_fc2 = nn.Linear(64, 32)
        
        #taking all the inputs from all the neurons and combining them to 64
        self.combined_input = nn.Linear(96, 64)  # 32+32+32 -> 64
        
        #takes 64 inputs (from the combined input) and makes it into a vector that has 12 outputs
        #one for each of the servo motors
        #calculates what the agent thinks the best action is
        self.mean_layer = nn.Linear(64, ACTION_SPACE_SIZE)
        
        #calculates how "sure" the agent is of the action
        self.log_std_layer = nn.Linear(64, ACTION_SPACE_SIZE)
        
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        def forward(self, imu, servo, lidar):
            x_imu = self.relu(self.imu_fc1(imu))
            x_imu = self.relu(self.imu_fc2(x_imu))
            
            x_servo = self.relu(self.servo_fc1(servo))
            x_servo = self.relu(self.servo_fc2(x_servo))

            x_lidar = self.relu(self.lidar_fc1(lidar))
            x_lidar = self.relu(self.lidar_fc2(x_lidar))
            
            combined = torch.cat([x_imu, x_servo, x_lidar], dim=-1)
            x = self.relu(self.combined_fc(combined))
            
            mean = self.tanh(self.mean_layer(x))
            log_std = self.tanh(self.log_std_layer(x))
            
            return mean, log_std

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.imu_fc1 = nn.Linear(6, 64)
        self.imu_fc2 = nn.Linear(64, 32)
        
        self.servo_fc1 = nn.Linear(12, 64)
        self.servo_fc2 = nn.Linear(64, 32)

        self.lidar_fc1 = nn.Linear(3, 64)
        self.lidar_fc2 = nn.Linear(64, 32)
        
        self.combined_fc = nn.Linear(96, 64)  # 32+32+32 -> 64
        
        self.value_layer = nn.Linear(64, 1)
        
        self.relu = nn.ReLU()
    
    def forward(self, imu, servo, lidar):
        x_imu = self.relu(self.imu_fc1(imu))
        x_imu = self.relu(self.imu_fc2(x_imu))
        
        x_servo = self.relu(self.servo_fc1(servo))
        x_servo = self.relu(self.servo_fc2(x_servo))
        
        x_lidar = self.relu(self.lidar_fc1(lidar))
        x_lidar = self.relu(self.lidar_fc2(x_lidar))
        
        combined = torch.cat([x_imu, x_servo, x_lidar], dim=-1)
        x = self.relu(self.combined_fc(combined))
        
        value = self.value_layer(x)
        
        return value
    