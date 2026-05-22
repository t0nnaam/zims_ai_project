from network import Actor, Critic
import random
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PPO:
    def __init__(self, discount=0.99, clipping=0.2, advantage=0.95, epoch=10, batch_size=64):
        # Networks - Map explicitly to Jetson's CUDA or CPU
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        
        # Hyperparameters
        self.discount = discount  
        self.clipping = clipping  
        self.advantage = advantage 
        self.epochs = epoch 
        self.batch_size = batch_size 
        self.max_grad_norm = 0.5 
        
        # Storage for trajectories
        self.states = {'imu': [], 'servo': [], 'lidar':[]}
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = [] 
        self.dones = []
    

    def choose_action(self, state):
        """ Choose action given observation """
        imu = state[:6]
        servo = state[6:18]
        lidar = state[18:]

        # Push feature tensors explicitly to device
        imu_tensor = torch.FloatTensor(imu).unsqueeze(0).to(device)
        servo_tensor = torch.FloatTensor(servo).unsqueeze(0).to(device)
        lidar_tensor = torch.FloatTensor(lidar).unsqueeze(0).to(device)
        
        # Get action from actor
        with torch.no_grad():
            action, log_prob = self.actor.get_action(imu_tensor, servo_tensor, lidar_tensor)
            value = self.critic.get_value(imu_tensor, servo_tensor, lidar_tensor)
        
        action_np = action.cpu().numpy().flatten()
        log_prob_np = log_prob.cpu().item()
        value_np = value.cpu().item()
    
        return action_np, log_prob_np, value_np


    def compute_log_prob(self, imu, servo, lidar, actions):
        imu_tensor = torch.FloatTensor(imu).to(device)
        servo_tensor = torch.FloatTensor(servo).to(device)
        lidar_tensor = torch.FloatTensor(lidar).to(device)
        actions_tensor = torch.FloatTensor(actions).to(device)
        
        mean, log_std = self.actor(imu_tensor, servo_tensor, lidar_tensor)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions_tensor).sum(dim=-1)
        return log_prob


    def save_model(self, filepath): 
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'hyperparameters': {
                'discount': self.discount,
                'clipping': self.clipping,
                'advantage': self.advantage,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'max_grad_norm': self.max_grad_norm
            }
        }
        torch.save(checkpoint, filepath)
    

    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


    def save_memory(self, obs, action, log_prob, value, reward, done):
        """ Store transition data in memory buffers """
        imu = obs[:6]
        servo = obs[6:18]
        lidar = obs[18:]
        
        self.states['imu'].append(imu)
        self.states['servo'].append(servo)
        self.states['lidar'].append(lidar)
        self.actions.append(action)
        self.rewards.append(reward)
        
        if isinstance(log_prob, np.ndarray):
            log_prob = log_prob.item()
        if isinstance(value, np.ndarray):
            value = value.item()
        
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)


    def rollout(self, env, num_steps):
        state = env.reset()
        episode_rewards = []
        episode_reward = 0
        episode_count = 0

        for step in range(num_steps):
            action, log_prob, value = self.choose_action(state)
            next_state, reward, done, info = env.step(action)
            self.save_memory(state, action, log_prob, value, reward, done)
            state = next_state
            episode_reward += reward

            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                print(f"Episode {episode_count} completed | Reward: {episode_reward:.2f}")
                state = env.reset()
                episode_reward = 0

        print(f"Collection complete! {num_steps} steps, {episode_count} episodes")
        if episode_rewards:
            print(f"Average reward: {np.mean(episode_rewards):.2f}")

        # FIX: Removed the non-existent self.get_data() call
        return episode_rewards
    

    def calcAdvantage(self, rewards, values, next_value, dones, gae_parameter = 0.95):
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)

        if isinstance(next_value, (int, float)):
            next_value = torch.tensor([next_value], dtype=torch.float32)
        elif next_value.dim() > 1:
            next_value = next_value.squeeze()

        advantages = [] 
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
                next_done = 0  
            else:
                next_val = values[t + 1]
                next_done = dones[t] 

            tdError = rewards[t] + self.discount * next_val * (1 - next_done) - values[t]
            gae = tdError + self.discount * gae_parameter * (1 - next_done) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages).to(device)


    def calcDiscountedReturns(self, rewards, dones):
        returns = []
        discounted_return = 0
        
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_return = 0
            discounted_return = reward + self.discount * discounted_return
            returns.insert(0, discounted_return)
        
        return torch.FloatTensor(returns).to(device)


    def update(self):
        """ update = learn function """
        if self.dones[-1]:
            next_value = 0.0
        else:
            # FIX: Extracted features straight from storage arrays, skipping non-existent _obs_to_tensors method
            imu_t = torch.FloatTensor(self.states['imu'][-1]).unsqueeze(0).to(device)
            servo_t = torch.FloatTensor(self.states['servo'][-1]).unsqueeze(0).to(device)
            lidar_t = torch.FloatTensor(self.states['lidar'][-1]).unsqueeze(0).to(device)
            with torch.no_grad():
                next_value = self.critic.get_value(imu_t, servo_t, lidar_t).item()

        advantages = self.calcAdvantage(rewards=self.rewards, values=self.values, next_value=next_value, dones=self.dones, gae_parameter=self.advantage)
        returns = self.calcDiscountedReturns(rewards=self.rewards, dones=self.dones)

        print(f"\n=== UPDATE DIAGNOSTICS ===")
        print(f"Advantages - Mean: {advantages.mean().item():.4f}")
        print(f"Returns - Mean: {returns.mean().item():.4f}")
        print(f"========================\n")

        old_values_np = np.array(self.values)       

        # Send target optimization data tensors straight to device
        imu_tensor = torch.FloatTensor(np.array(self.states['imu'])).to(device)
        servo_tensor = torch.FloatTensor(np.array(self.states['servo'])).to(device)
        lidar_tensor = torch.FloatTensor(np.array(self.states['lidar'])).to(device)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(device)
        old_log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).squeeze().to(device)
        old_values_tensor = torch.FloatTensor(old_values_np).to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        num_samples = len(self.rewards)

        for epoch in range(self.epochs):
            indices = torch.randperm(num_samples)
            
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_indices = indices[start:end]
                
                self.updateActor(
                    imu_tensor[batch_indices], servo_tensor[batch_indices], lidar_tensor[batch_indices],
                    actions_tensor[batch_indices], old_log_probs_tensor[batch_indices], advantages[batch_indices]
                )
                
                self.updateCritic(
                    imu_tensor[batch_indices], servo_tensor[batch_indices], lidar_tensor[batch_indices],
                    returns[batch_indices], old_values_tensor[batch_indices]
                )  

        self.clear_memory()
    

    def updateActor(self, imu, servo, lidar, actions,
