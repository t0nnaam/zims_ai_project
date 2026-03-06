
import torch
import torch.nn as nn
import numpy as np

ACTION_SPACE_SIZE = 12

#Actor state input consists of IMU (6), Servo (12) if Lidar is added (3)

# I changed the NN to use pytorch instead of tensorflow cuz that's what was covered last session
class Actor(nn.Module): 
    def __init__(self):
        super(Actor, self).__init__()
        
        self.imu_fc1 = nn.Linear(6, 64)
        self.imu_fc2 = nn.Linear(64, 32)
        
        self.servo_fc1 = nn.Linear(12, 64)
        self.servo_fc2 = nn.Linear(64, 32)
        
        self.lidar_fc1 = nn.Linear(3, 64)
        self.lidar_fc2 = nn.Linear(64, 32)
        
        self.combined_fc = nn.Linear(96, 64)
        
        self.mean_layer = nn.Linear(64, ACTION_SPACE_SIZE)
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
    
    def get_action(self, state):
        # Split state into components
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        imu = state[:, :6]
        servo = state[:, 6:18]
        lidar = state[:, 18:21]
        
        # Get mean and log_std from forward pass
        mean, log_std = self.forward(imu, servo, lidar)
        std = log_std.exp()
        
        # Create normal distribution and sample
        from torch.distributions import Normal
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        
        self.imu_fc1 = nn.Linear(6, 64)
        self.imu_fc2 = nn.Linear(64, 32)
        
        self.servo_fc1 = nn.Linear(12, 64)
        self.servo_fc2 = nn.Linear(64, 32)

        self.lidar_fc1 = nn.Linear(3, 64)
        self.lidar_fc2 = nn.Linear(64, 32)
        
        self.combined_fc = nn.Linear(96, 64)
        
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
    
    def get_value(self, state):
        # Split state into components
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        imu = state[:, :6]
        servo = state[:, 6:18]
        lidar = state[:, 18:21]
        
        value = self.forward(imu, servo, lidar)
        
        return value

    def getValues(self, imu, servo, lidar): 
        with torch.no_grad(): 
            value = self.forward(imu, servo, lidar) 

        return value.detach().squeeze()
    