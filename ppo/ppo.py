# from import *
from network import Actor, Critic
import random
import torch
import torch.nn as nn
import numpy as np

class PPO:
    def __init__(self, discount=0.99, clipping=0.2, advantage=0.95, epoch=10, batch_size=64):
        # Networks
        self.actor = Actor()
        self.critic = Critic()
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0003)
        
        # Hyperparameters
        self.discount = discount  # discount factor
        self.clipping = clipping  # PPO clipping parameter
        self.advantage = advantage # for advantage estimation
        self.epochs = epoch # number of epochs
        self.batch_size = batch_size # number of samples per update
        self.max_grad_norm = 0.5 # safety limit for gradient magnitude to prevent exploding gradients
        
        # Storage for trajectories
        self.states = {'imu': [], 'servo': [], 'lidar':[]}
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = [] # store old log probs for PPO
        self.dones = []
    

    def choose_action(self, state):
        """
        Choose action given observation
        """
        #split each sensor into separate paramaters for get_action and get_value
        imu = state[:6]
        servo = state[6:18]
        lidar = state[18:]

        # make each one into a tensor
        imu_tensor = torch.FloatTensor(imu).unsqueeze(0)
        servo_tensor = torch.FloatTensor(servo).unsqueeze(0)
        lidar_tensor = torch.FloatTensor(lidar).unsqueeze(0)
        
        # Get action from actor
        with torch.no_grad():
            action, log_prob = self.actor.get_action(imu_tensor, servo_tensor, lidar_tensor)
            value = self.critic.get_value(imu_tensor, servo_tensor, lidar_tensor)
        
        # Convert to numpy for environment
        action_np = action.cpu().numpy().flatten()
        log_prob_np = log_prob.cpu().item()
        value_np = value.cpu().item()
    
        return action_np, log_prob_np, value_np


    def compute_log_prob(self, imu, servo, lidar, actions):
        # Convert states to tensors if not already
        imu_tensor = torch.FloatTensor(imu)
        servo_tensor = torch.FloatTensor(servo)
        lidar_tensor = torch.FloatTensor(lidar)
        actions_tensor = torch.FloatTensor(actions)
        
        # Forward pass through actor
        mean, log_std = self.actor(imu_tensor, servo_tensor, lidar_tensor)
        std = log_std.exp()
        
        # Create the distribution
        dist = torch.distributions.Normal(mean, std)
        
        # Compute log probability of the taken actions
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
        """
        Load actor and critic + optimizer states
        """
        checkpoint = torch.load(filepath)

        # Load network parameters
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        
        # Load optimizer states
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


    def save_memory(self, obs, action, log_prob, value, reward, done):
        """
        Store transition data in memory buffers
        """
        # Split observation into components
        imu = obs[:6]
        servo = obs[6:18]
        lidar = obs[18:]
        
        # Store each component
        self.states['imu'].append(imu)
        self.states['servo'].append(servo)
        self.states['lidar'].append(lidar)
        self.actions.append(action)
        self.rewards.append(reward)
        
        # Convert log_prob and value to scalars if they're arrays
        if isinstance(log_prob, np.ndarray):
            log_prob = log_prob.item()
        if isinstance(value, np.ndarray):
            value = value.item()
        
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)


    def rollout(self, env, num_steps):
        # Reset environment to get initial state
        state = env.reset()

        # Statistics tracking
        episode_rewards = []
        episode_reward = 0
        episode_count = 0

        # Collect num_steps of data
        for step in range(num_steps):
            action, log_prob, value = self.choose_action(state)
            next_state, reward, done, info = env.step(action)
            self.save_memory(state, action, log_prob, value, reward, done)
            state = next_state
            episode_reward += reward

            # Reset environment if episode is done
            if done:
                episode_count += 1
                episode_rewards.append(episode_reward)
                print(f"Episode {episode_count} completed | Reward: {episode_reward:.2f}")

                state = env.reset()
                episode_reward = 0

        # print collection summary
        print(f"Collection complete! {num_steps} steps, {episode_count} episodes")
        if episode_rewards:
            print(f"Average reward: {np.mean(episode_rewards):.2f}")

        # return collected data
        return self.get_data()
    


    def calculateTDResidual(self, rewards, values, next_value, dones): 
        # Convert to tensors
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)

        # Handle next_value - ensure it's a tensor
        if isinstance(next_value, (int, float)):
            next_value = torch.tensor([next_value], dtype=torch.float32)
        elif next_value.dim() > 1:
            next_value = next_value.squeeze()

        next_values = torch.cat([values[1:], next_value])

        mask_tensor = 1.0 - dones

        td_residual = rewards + (self.discount * mask_tensor * next_values) - values
        
        return td_residual 


    def calcAdvantage(self, rewards, values, next_value, dones, gae_parameter = 0.95):
        """ Calculate Advantage Estimation """
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)

        # Handle next_value
        if isinstance(next_value, (int, float)):
            next_value = torch.tensor([next_value], dtype=torch.float32)
        elif next_value.dim() > 1:
            next_value = next_value.squeeze()

        # if done: 
        advantages = [] 
        gae = 0;
        
        # Calculate advantages backwards through the trajectory
        for t in reversed(range(len(rewards))):
            # Determine next value for this timestep
            if t == len(rewards) - 1:
                next_val = next_value
                next_done = 0  # Assume not done for bootstrap
            else:
                next_val = values[t + 1]
                next_done = dones[t] # is it t or t+1

            tdError = rewards[t] + self.discount * next_val * (1 - next_done) - values[t]
            
            gae = tdError + self.discount * gae_parameter * (1 - next_done) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages)


    def calcDiscountedReturns(self, rewards, dones):
        """ Calculate Discounted Returns """
        returns = []
        discounted_return = 0
        
        # Calculate returns backwards through the trajectory
        for reward, done in zip(reversed(rewards), reversed(dones)):
            # Reset return to 0 if episode ended
            if done:
                discounted_return = 0
        
            discounted_return = reward + self.discount * discounted_return
            returns.insert(0, discounted_return)
        
        return torch.FloatTensor(returns)


    def update(self):
        """ 
        update = learn function
        Calculate advantages, create minibatches, call actor and critic updates 
        """
        if self.dones[-1]:
            next_value = 0.0
        # otherwise estimate from critic    
        else:
            # Get the last state and estimate its value
            last_obs = np.concatenate([self.states['imu'][-1], self.states['servo'][-1], self.states['lidar'][-1]])
            imu = last_obs[:6]
            servo = last_obs[6:18]
            lidar = last_obs[18:]
            imu_t = torch.FloatTensor(imu).unsqueeze(0)
            servo_t = torch.FloatTensor(servo).unsqueeze(0)
            lidar_t = torch.FloatTensor(lidar).unsqueeze(0)
            with torch.no_grad():
                next_value = self.critic.get_value(imu_t, servo_t, lidar_t).item()

        advantages = self.calcAdvantage(rewards=self.rewards, values=self.values, next_value=next_value, dones=self.dones, gae_parameter=self.advantage)
        returns = self.calcDiscountedReturns(rewards=self.rewards, dones=self.dones)

        print(f"\n=== UPDATE DIAGNOSTICS ===")
        print(f"Advantages - Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")
        print(f"Returns - Mean: {returns.mean():.4f}, Std: {returns.std():.4f}")
        print(f"Values - Mean: {np.mean(self.values):.4f}, Std: {np.std(self.values):.4f}")
        print(f"Rewards - Mean: {np.mean(self.rewards):.4f}, Std: {np.std(self.rewards):.4f}")
        print(f"========================\n")

         # Store old values before converting to tensor
        old_values_np = np.array(self.values)       

        # Convert data to tensor
        imu_tensor = torch.FloatTensor(np.array(self.states['imu']))
        servo_tensor = torch.FloatTensor(np.array(self.states['servo']))
        lidar_tensor = torch.FloatTensor(np.array(self.states['lidar']))
        actions_tensor = torch.FloatTensor(np.array(self.actions))
        old_log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).squeeze()
        old_values_tensor = torch.FloatTensor(old_values_np)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get number of samples
        num_samples = len(self.rewards)

        # Update for multiple epochs
        for epoch in range(self.epochs):
            # Create random minibatches
            indices = torch.randperm(num_samples)
            
            for start in range(0, num_samples, self.batch_size):
                end = min(start + self.batch_size, num_samples)
                batch_indices = indices[start:end]
                
                # Create minibatch
                batch_imu = imu_tensor[batch_indices]
                batch_servo = servo_tensor[batch_indices]
                batch_lidar = lidar_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values_tensor[batch_indices]
                
                # Update actor
                self.updateActor(batch_imu, batch_servo, batch_lidar, batch_actions, batch_old_log_probs, batch_advantages)
                
                # Update critic
                self.updateCritic(batch_imu, batch_servo, batch_lidar, batch_returns, batch_old_values)  

        self.clear_memory()
    

    def updateActor(self, imu, servo, lidar, actions, log_prob_old, advantages):
        log_prob_new = self.compute_log_prob(imu, servo, lidar, actions)
        
        # Calculate probability ratio = pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        ratio = torch.exp(log_prob_new - log_prob_old)
    
        # Calculate clipped vs unclipped loss
        unclippedL = ratio * advantages
        ratio_clipped  = torch.clamp(ratio, 1 - self.clipping, 1 + self.clipping)
        clippedL = ratio_clipped  * advantages

        # Entropy bonus for exploration
        mean, log_std = self.actor(imu, servo, lidar)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        entropy = dist.entropy().sum(dim=-1).mean()

        # actor loss = pessimistic estimate, take minimum = policy loss - entropy bonus
        actor_loss = -torch.min(unclippedL, clippedL).mean() - 0.01 * entropy

        # Calculate gradients, backward propagation for actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()

        return actor_loss.item()
    

    def updateCritic(self, imu, servo, lidar, returns, old_values):
        # Ensure returns has the correct shape
        if returns.dim() == 1:
            returns = returns.unsqueeze(-1)

        # Forward Pass: predict state values
        predicted_values = self.critic(imu, servo, lidar)

        old_values = old_values.unsqueeze(-1) if old_values.dim() == 1 else old_values
        value_pred_clipped = old_values + torch.clamp(predicted_values - old_values, -self.clipping, self.clipping)

        value_loss_unclipped = (predicted_values - returns).pow(2)
        value_loss_clipped = (value_pred_clipped - returns).pow(2)
    
        loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

        self.critic_optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

        self.critic_optimizer.step()

        return loss.item()
    

    def clear_memory(self):
        self.states = {'imu': [], 'servo': [], 'lidar': []}
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        print("Buffers cleared")