import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

ACTION_SPACE_SIZE = 12

class Actor(nn.Module): 
    def __init__(self):
        super(Actor, self).__init__()
        
        # IMU processing (6 inputs -> 32 outputs)
        self.imu_fc1 = nn.Linear(6, 64)
        self.imu_fc2 = nn.Linear(64, 32)
        
        # Servo processing (12 inputs -> 32 outputs)
        self.servo_fc1 = nn.Linear(12, 64)
        self.servo_fc2 = nn.Linear(64, 32)
        
        # Lidar processing (Changed to 360 for standard 1-degree increments)
        self.lidar_fc1 = nn.Linear(360, 128)
        self.lidar_fc2 = nn.Linear(128, 64)
        
        # Combined sensor feature space (32 + 32 + 64 = 128)
        self.combined_fc = nn.Linear(128, 64)
        
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
        
        # FIX: Removed Tanh to allow exploration variance to shrink down cleanly
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2) 
        
        return mean, log_std
    
    def get_action(self, imu, servo, lidar):
        mean, log_std = self.forward(imu, servo, lidar)
        std = log_std.exp()  
        dist = torch.distributions.Normal(mean, std)
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

        self.lidar_fc1 = nn.Linear(360, 128)
        self.lidar_fc2 = nn.Linear(128, 64)
        
        self.combined_fc = nn.Linear(128, 64)
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
    
    def get_value(self, imu, servo, lidar):
        with torch.no_grad():  
            return self.forward(imu, servo, lidar)
