# Spider Robot PPO Reinforcement Learning

A PyTorch implementation of Proximal Policy Optimization (PPO) for training a quadruped spider robot to walk - for zotbotics ai project. 

This project trains a spider robot with 4 legs (12 servo motors total - 3 per leg) to navigate and walk using reinforcement learning. The robot uses:
- **IMU sensors** (6 values) - orientation and acceleration data
- **Servo positions** (12 values) - current joint angles
- **LIDAR** (3 values) - obstacle detection

The PPO agent learns to control all 12 servos simultaneously to produce stable walking gaits.

### Prerequisites
- Python 3.8+
- PyTorch
- NumPy
- PyBullet (for physics simulation)
- Gymnasium (for environment interface)

### Install Dependencies

```bash
pip install torch numpy pybullet gymnasium
```

## How to Run

### 1. Train the Model

Start training from scratch:

```bash
python run_env.py
```

The script will:
- Initialize the spider robot environment
- Train the PPO agent for 1,000,000 steps
- Save checkpoints every 10 episodes to `./models/spider_ppo.pth`
- Log training metrics to `reward_log.txt`

### 2. Resume Training

To continue training from a saved checkpoint:

```python
# In run_env.py, set:
RESUME_TRAINING = True
```

Then run:
```bash
python run_env.py
```

### 3. Evaluate/Run Without Training

To watch the trained robot without further training:

```python
# In run_env.py, set:
train = False
```

Then run:
```bash
python run_env.py
```

## Configuration

Edit `run_env.py` to adjust training parameters:

```python
TOTAL_STEPS   = 1_000_000   # Total training steps
SAVE_INTERVAL = 10          # Save model every N episodes
RESUME_TRAINING = True      # Resume from checkpoint
train = True                # Enable/disable training
```

Edit `ppo.py` to adjust PPO hyperparameters:

```python
PPO(
    discount=0.99,      # Gamma - discount factor
    clipping=0.2,       # Epsilon - PPO clip range
    advantage=0.95,     # Lambda - GAE parameter
    epoch=10,           # Update epochs per rollout
    batch_size=64       # Minibatch size
)
```
