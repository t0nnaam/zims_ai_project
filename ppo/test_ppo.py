import torch
import numpy as np
from ppo import PPO

# Shout out Claude for this test code
# now we just pray its correct

# Mock environment for testing
class MockSpiderEnv:
    def __init__(self):
        self.state_dim = 21  # 6 IMU + 12 servo + 3 lidar
        self.action_dim = 12
        self.max_steps = 100
        self.current_step = 0
        
    def reset(self):
        """Reset environment and return initial state"""
        self.current_step = 0
        # Return random state: [imu(6), servo(12), lidar(3)]
        return np.random.randn(self.state_dim).astype(np.float32)
    
    def step(self, action):
        """Execute action and return next_state, reward, done, info"""
        self.current_step += 1
        
        # Next state
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        
        # Simple reward (replace with actual spider robot reward)
        reward = -np.sum(action**2) * 0.01 + np.random.randn() * 0.1
        
        # Done flag
        done = self.current_step >= self.max_steps or np.random.rand() < 0.02
        
        # Info
        info = {}
        
        return next_state, reward, done, info


# Test 1: Test PPO initialization
def test_initialization():
    print("=" * 50)
    print("TEST 1: PPO Initialization")
    print("=" * 50)
    
    try:
        ppo = PPO(
            discount=0.99,
            clipping=0.2,
            advantage=0.95,
            epoch=10,
            batch_size=64
        )
        print("✓ PPO initialized successfully")
        print(f"  - Actor network: {ppo.actor}")
        print(f"  - Critic network: {ppo.critic}")
        print(f"  - Hyperparameters: discount={ppo.discount}, clipping={ppo.clipping}")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


# Test 2: Test rollout collection
def test_rollout():
    print("\n" + "=" * 50)
    print("TEST 2: Rollout Collection")
    print("=" * 50)
    
    try:
        ppo = PPO()
        env = MockSpiderEnv()
        
        # Collect 100 steps
        print("Collecting 100 steps...")
        data = ppo.rollout(env, num_steps=100)
        
        print(f"✓ Rollout completed successfully")
        # FIXED: Access nested dictionary structure
        print(f"  - IMU states collected: {data['states']['imu'].shape}")
        print(f"  - Servo states collected: {data['states']['servo'].shape}")
        print(f"  - Lidar states collected: {data['states']['lidar'].shape}")
        print(f"  - Actions collected: {data['actions'].shape}")
        print(f"  - Rewards collected: {data['rewards'].shape}")
        print(f"  - Log probs collected: {data['log_probs'].shape}")
        print(f"  - Values collected: {data['values'].shape}")
        print(f"  - Dones collected: {data['dones'].shape}")
        print(f"  - Average reward: {np.mean(data['rewards']):.4f}")
        
        # FIXED: Check each component of states
        assert len(data['states']['imu']) == 100, "IMU states count mismatch"
        assert len(data['states']['servo']) == 100, "Servo states count mismatch"
        assert len(data['states']['lidar']) == 100, "Lidar states count mismatch"
        assert len(data['actions']) == 100, "Actions count mismatch"
        assert len(data['rewards']) == 100, "Rewards count mismatch"
        
        return True
    except Exception as e:
        print(f"✗ Rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 3: Test advantage calculation
def test_advantage_calculation():
    print("\n" + "=" * 50)
    print("TEST 3: Advantage Calculation")
    print("=" * 50)
    
    try:
        ppo = PPO()
        
        # Create dummy data
        ppo.rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        ppo.rewards_tensor = torch.tensor(ppo.rewards, dtype=torch.float32)
        
        dummy_values = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5], dtype=torch.float32)
        ppo.critic_values_tensor = dummy_values
        
        # Test TD residual calculation
        next_value = torch.tensor([5.5], dtype=torch.float32)
        done = False
        
        td_residual = ppo.calculateTDResidual(next_value, done)
        print(f"✓ TD Residual calculated: {td_residual}")
        print(f"  - Shape: {td_residual.shape}")
        
        # Test advantage calculation
        advantage = ppo.calcAdvantage(next_value, next_advantage=0.0, done=False)
        print(f"✓ Advantage calculated: {advantage}")
        print(f"  - Shape: {advantage.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Advantage calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 4: Test discounted returns
def test_discounted_returns():
    print("\n" + "=" * 50)
    print("TEST 4: Discounted Returns")
    print("=" * 50)
    
    try:
        ppo = PPO(discount=0.99)
        
        # Create dummy rewards
        ppo.rewards = [1.0, 1.0, 1.0, 1.0, 1.0]
        ppo.rewards_tensor = torch.tensor(ppo.rewards, dtype=torch.float32)
        
        returns = ppo.calcDiscountedReturns()
        print(f"✓ Discounted returns calculated: {returns}")
        print(f"  - Shape: {returns.shape}")
        print(f"  - Values: {returns.numpy()}")
        
        # Check if returns are decreasing (due to discounting)
        assert returns[0] > returns[-1], "Returns should decrease with discount"
        
        return True
    except Exception as e:
        print(f"✗ Discounted returns failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 5: Test actor forward pass
def test_actor_forward():
    print("\n" + "=" * 50)
    print("TEST 5: Actor Forward Pass")
    print("=" * 50)
    
    try:
        from network import Actor
        
        actor = Actor()
        
        # Create dummy inputs
        batch_size = 4
        imu = torch.randn(batch_size, 6)
        servo = torch.randn(batch_size, 12)
        lidar = torch.randn(batch_size, 3)
        
        # Forward pass
        mean, log_std = actor(imu, servo, lidar)
        
        print(f"✓ Actor forward pass successful")
        print(f"  - Mean shape: {mean.shape}")
        print(f"  - Log_std shape: {log_std.shape}")
        print(f"  - Mean sample: {mean[0]}")
        print(f"  - Log_std sample: {log_std[0]}")
        
        return True
    except Exception as e:
        print(f"✗ Actor forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 6: Test critic forward pass
def test_critic_forward():
    print("\n" + "=" * 50)
    print("TEST 6: Critic Forward Pass")
    print("=" * 50)
    
    try:
        from network import Critic
        
        critic = Critic()
        
        # Create dummy inputs
        batch_size = 4
        imu = torch.randn(batch_size, 6)
        servo = torch.randn(batch_size, 12)
        lidar = torch.randn(batch_size, 3)
        
        # Forward pass
        value = critic(imu, servo, lidar)
        
        print(f"✓ Critic forward pass successful")
        print(f"  - Value shape: {value.shape}")
        print(f"  - Value sample: {value[0]}")
        
        return True
    except Exception as e:
        print(f"✗ Critic forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 7: Test memory clearing
def test_clear_memory():
    print("\n" + "=" * 50)
    print("TEST 7: Clear Memory")
    print("=" * 50)
    
    try:
        ppo = PPO()
        env = MockSpiderEnv()
        
        # Collect some data
        ppo.rollout(env, num_steps=50)
        
        print(f"Before clear: {len(ppo.states)} states")
        
        # Clear memory
        ppo.clear_memory()
        
        print(f"After clear: {len(ppo.states)} states")
        
        assert len(ppo.states) == 0, "States should be empty"
        assert len(ppo.actions) == 0, "Actions should be empty"
        assert len(ppo.rewards) == 0, "Rewards should be empty"
        
        print("✓ Memory cleared successfully")
        
        return True
    except Exception as e:
        print(f"✗ Clear memory failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test 8: Test full training step
def test_update_step():
    print("\n" + "=" * 50)
    print("TEST 8: Update Step")
    print("=" * 50)
    
    try:
        ppo = PPO(discount=0.99, clipping=0.2, epoch=3, batch_size=32)
        env = MockSpiderEnv()
        
        # Collect trajectory data
        print("Collecting trajectory data...")
        ppo.rollout(env, num_steps=128)
        
        # Store initial network parameters to verify they change
        initial_actor_params = [p.clone() for p in ppo.actor.parameters()]
        initial_critic_params = [p.clone() for p in ppo.critic.parameters()]
        
        print(f"Data collected: {len(ppo.rewards)} steps")
        print(f"Actor parameters before update: {len(list(ppo.actor.parameters()))}")
        print(f"Critic parameters before update: {len(list(ppo.critic.parameters()))}")
        
        # Perform update
        print("\nPerforming PPO update...")
        ppo.update()
        
        # Check that parameters actually changed
        actor_changed = False
        for initial, current in zip(initial_actor_params, ppo.actor.parameters()):
            if not torch.equal(initial, current):
                actor_changed = True
                break
        
        critic_changed = False
        for initial, current in zip(initial_critic_params, ppo.critic.parameters()):
            if not torch.equal(initial, current):
                critic_changed = True
                break
        
        print(f"✓ Update completed successfully")
        print(f"  - Actor parameters changed: {actor_changed}")
        print(f"  - Critic parameters changed: {critic_changed}")
        print(f"  - Memory cleared: {len(ppo.states['imu']) == 0}")
        
        # Verify memory was cleared
        assert len(ppo.states['imu']) == 0, "States should be cleared after update"
        assert len(ppo.actions) == 0, "Actions should be cleared after update"
        assert len(ppo.rewards) == 0, "Rewards should be cleared after update"
        
        # Verify networks updated
        assert actor_changed, "Actor parameters should change after update"
        assert critic_changed, "Critic parameters should change after update"
        
        print("✓ All update checks passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Update step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# Run all tests
def run_all_tests():
    print("\n" + "=" * 70)
    print(" " * 20 + "PPO TESTING SUITE")
    print("=" * 70)
    
    results = {
        "Initialization": test_initialization(),
        "Rollout Collection": test_rollout(),
        "Advantage Calculation": test_advantage_calculation(),
        "Discounted Returns": test_discounted_returns(),
        "Actor Forward": test_actor_forward(),
        "Critic Forward": test_critic_forward(),
        "Clear Memory": test_clear_memory(),
        "Update Step": test_update_step(),
    }
    
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v == True)
    failed = sum(1 for v in results.values() if v == False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result == True else ("✗ FAILED" if result == False else "⚠ SKIPPED")
        print(f"{test_name:.<40} {status}")
    
    print("=" * 70)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()