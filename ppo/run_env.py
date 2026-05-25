import os
import numpy as np
import time
import board
import busio
from adafruit_servokit import ServoKit
from sensors import LidarReader, IMUReader
from ppo import PPO

# ── Hardware Setup ────────────────────────────────────────────────────────────
# Initialize your real I2C bus and Servo driver tested in your screenshot
i2c = busio.I2C(board.SCL, board.SDA)
kit = ServoKit(channels=16, i2c=i2c)

lidar = LidarReader(port='/dev/ttyUSB0')
lidar.connect()
print("LiDAR connected.")
 
imu = IMUReader(bus=1)
print("IMU connected.")

MODEL_PATH = "./models/spider_ppo.pth"

# ── Physical Robot Environment Wrapper ────────────────────────────────────────
class RealSpiderEnv:
    def __init__(self):
        # Initialize your physical sensors here (e.g., Mini LiDAR, IMU)
        pass

    def reset(self):
        print("Resetting robot to default standing pose...")
        # Command your real servos to a starting position
        for i in range(12):
            kit.servo[i].angle = 90 # Adjust to your robot's calibration midpoint
        time.sleep(1.0)
        return self._get_hardware_observations()

    def _get_hardware_observations(self):
        # 1. Read your real 6-axis IMU values (accel x/y/z, gyro r/p/y)
        imu_data = [0.0] * 6 # Replace with your real IMU sensor read library
        # imu_data = imu.get_reading()
        
        # 2. Get your current 12 servo angles
        servo_data = [kit.servo[i].angle for i in range(12)]
        
        # 3. Read your 360-degree LiDAR array
        # lidar_data = [1.0] * 360 # Replace with your real LiDAR serial read library
        lidar_data = lidar.get_scan().tolist()
        
        # Combine them into a single state vector matching your network's expectations
        return np.concatenate([imu_data, servo_data, lidar_data])

    def step(self, action):
        # Map the continuous PPO action outputs (usually between -1 and 1) to physical angles (0 to 180)
        for i in range(12):
            target_angle = int((action[i] + 1) * 90) # Maps -1 -> 0 deg, 0 -> 90 deg, 1 -> 180 deg
            # Clip values safely to ensure servos don't force-jam your linkages
            target_angle = max(20, min(160, target_angle)) 
            kit.servo[i].angle = target_angle
        
        # Give the hardware a tiny fraction of a second to physically move
        time.sleep(0.05) 
        
        next_obs = self._get_hardware_observations()

        # Fall detection via IMU — if tilt is severe, end the episode
        ax, ay, az = next_obs[0], next_obs[1], next_obs[2]
        fallen     = abs(ax) > 8.0 or abs(ay) > 8.0   # ~80% tilt in m/s²
        reward     = 1.0 if not fallen else -10.0
        
        # Calculate a safety reward (e.g., negative penalty if the IMU detects it fell over)
        reward = 1.0 
        # terminated = False # Set to True if your IMU detects a catastrophic tilt/fall
        # Fall detection via IMU — if tilt is severe, end the episode
        ax, ay, az = next_obs[0], next_obs[1], next_obs[2]
        fallen     = abs(ax) > 8.0 or abs(ay) > 8.0   # ~80% tilt in m/s²
        reward     = 1.0 if not fallen else -10.0
        info = {}
        
        return next_obs, reward, fallen, truncated, info

# ── Run Deployment ────────────────────────────────────────────────────────────
env = RealSpiderEnv()
agent = PPO()

# Load the brain weights trained in your simulation setup
if os.path.exists(MODEL_PATH):
    agent.load_model(MODEL_PATH)
    print("Successfully loaded simulation weights! Running live hardware deployment...")
else:
    print(f"Warning: Checkpoint not found at {MODEL_PATH}. Running with randomized weights.")

obs = env.reset()

print("Robot active. Press Ctrl+C to stop.")
try:
    while True:
        # Pass live hardware sensor readings to the neural network
        action, log_prob, value = agent.choose_action(obs)
        
        # Exert actions to physical servos and gather the next state
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs = env.reset()

except KeyboardInterrupt:
    print("\nShutting down safely. Relaxing all servos.")
    # Optional: Turn off servo signals so they don't draw continuous power/heat while standing still
    for i in range(12):
        kit.servo[i].angle = None
