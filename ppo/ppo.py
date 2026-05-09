# from import *
from network import Actor, Critic
import random
import torch
import torch.nn as nn
import numpy as np

# TODO 
# Bug fix (from test cases): update (learn) function, Advantage Calculation
# If you can, download pybullet and gymnasium so you can run the test environment

class PPO:
    def __init__(self, discount=0.99, clipping=0.2, advantage=0.9, epoch=10, batch_size=64):
        # the values here are placeholders rn - we can test and change later
        # Networks
        self.actor = Actor()
        self.critic = Critic()
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        
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
        # self.rewards_tensor = torch.tensor(self.rewards, dtype = torch.float32)
        # self.critic_values_tensor = self.critic.getValues(self.states['imu'], self.states['servo'], self.states['lidar']).detach()


    def choose_action(self, state):
        """
        Choose action given observation
        """
        #split each sensor into separate paramaters for get_action and get_value
        imu = state[:6]
        servo = state[6:18]
        lidar = state[18:21]

        # make each one into a tensor
        imu_tensor = torch.FloatTensor(imu).unsqueeze(0)
        servo_tensor = torch.FloatTensor(servo).unsqueeze(0)
        lidar_tensor = torch.FloatTensor(lidar).unsqueeze(0)
        
        # Get action from actor
        with torch.no_grad():
            # use Actor to sample action (actor's distribution)
            action, log_prob = self.actor.get_action(imu_tensor, servo_tensor, lidar_tensor)
            # Get value estimate from critic
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
        """
        Save actor and critic + optimizer states
        """
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
        #actor.save_checkpoint
        #critic.save_checkpoint
    

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
        
        # do we need to load hyperparameters?

        #actor.load_checkpoint
        #critic.load_checkpoint


    def save_memory(self, obs, action, log_prob, value, reward, done):
        """
        Store transition data in memory buffers
        """
        # Split observation into components
        imu = obs[:6]
        servo = obs[6:18]
        lidar = obs[18:21]
        
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
        """
        from zihanwang126-branch
        Run policy in environment and collect num_steps of data

        Args:
            env: Environment to interact with
            num_steps: Number of steps to collect

        Returns:
            Dictionary containing collected trajectory data
        """
        print(f"\nStarting trajectory collection for {num_steps} steps")

        # Reset environment to get initial state
        state = env.reset()

        # Statistics tracking
        episode_rewards = []
        episode_reward = 0
        episode_count = 0

        # Note: A lot of the stuff here was moved to the functions we were told to write in office hours
        # Collect num_steps of data
        for step in range(num_steps):
            # Use choose_action to get action, log_prob, and value 
            action, log_prob, value = self.choose_action(state)
    
            # Convert state to tensor format
            # state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Execute action in environment
            next_state, reward, done, info = env.step(action)

            # Store transition data using save_memory
            self.save_memory(state, action, log_prob, value, reward, done)

            # Update state and statistics
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


    def get_data(self):
        """
        helper function from zihanwang126-branch
        Get collected data and convert to numpy arrays

        Returns:
            Dictionary containing all trajectory data
        """
        data = {
            'states': {
                'imu': np.array(self.states['imu']),
                'servo': np.array(self.states['servo']),
                'lidar': np.array(self.states['lidar'])
            },
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values).flatten(),
            'dones': np.array(self.dones)
        }
        return data


    def calculateTDResidual(self, next_value, rewards, values, dones): 
        # Convert to tensors
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)

        # Handle next_value - ensure it's a tensor
        if isinstance(next_value, (int, float)):
            next_value = torch.tensor([next_value], dtype=torch.float32)
        elif next_value.dim() > 1:
            next_value = next_value.squeeze()

        # Build next_values
        # next_values = torch.cat([critic_values[1:], next_value.unsqueeze(0)]) 
        next_values = torch.cat([values[1:], next_value])

        # Mask for non-terminal states (1 = not done, 0 = done)
        mask_tensor = 1.0 - dones

        # uses vectorization to calculate the TD residual 
        # current reward + (discount factor * next critic value) - current critic value 
        td_residual = rewards + (self.discount * mask_tensor * next_values) - values
        
        return td_residual 


    def calcAdvantage(self, rewards, values, next_value, dones, gae_parameter = 0.95):
        """ Calculate Advantage Estimation """
        # Convert to tensors
        rewards = torch.FloatTensor(rewards)
        values = torch.FloatTensor(values)
        dones = torch.FloatTensor(dones)
        # rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)
        # dones_tensor = torch.tensor(self.dones, dtype=torch.float32)
        # critic_values_tensor = torch.tensor(self.values, dtype=torch.float32).flatten()

        # Handle next_value
        if isinstance(next_value, (int, float)):
            next_value = torch.tensor([next_value], dtype=torch.float32)
        elif next_value.dim() > 1:
            next_value = next_value.squeeze()

        # if done: 
        advantages = [] = 0 
        
        # Calculate advantages backwards through the trajectory
        for t in reversed(range(len(rewards))):
            # Determine next value for this timestep
            if t == len(rewards) - 1:
                next_val = next_value
                next_done = 0  # Assume not done for bootstrap
            else:
                next_val = values[t + 1]
                next_done = dones[t]
            
            # return self.calculateTDResidual(next_value, rewards_tensor, critic_values_tensor, dones_tensor) + (self.discount * gae_parameter * next_advantage)
            # TD error = current reward + (discount * next value * not_done) - current value
            tdError = rewards[t] + self.discount * next_val * (1 - next_done) - values[t]
            
            # GAE (advantage) = TD error + (discount * gae_lambda * not_done * previous advantage)
            gae = tdError + self.discount * gae_parameter * (1 - next_done) * gae
            advantages.insert(0, gae)

            return torch.FloatTensor(advantages)


    def calcDiscountedReturns(self, rewards, dones):
        """ Calculate Discounted Returns """
	    # (discounted factor ^ range from 0 to the number of rows of the rewards tensor) * the rewards tensor
        # Return = current reward + (discount * next return)
        returns = []
        discounted_return = 0
        
        # Calculate returns backwards through the trajectory
        for reward, done in zip(reversed(rewards), reversed(dones)):
            # Reset return to 0 if episode ended
            if done:
                discounted_return = 0
            
            # Return = current reward + (discount * next return)
            discounted_return = reward + self.discount * discounted_return
            returns.insert(0, discounted_return)
        
        # return self.discount ** torch.arange(self.rewards_tensor.size(0)) * self.rewards_tensor
        return torch.FloatTensor(returns)


    # update = learn function
    def update(self):
        """ 
        update = learn function
        Calculate advantages, create minibatches, call actor and critic updates 
        """
        # get next value
        # if last state was terminal return 0
        if self.dones[-1]:
            next_value = 0.0
        # otherwise estimate from critic    
        else:
            # Get the last state and estimate its value
            last_obs = np.concatenate([self.states['imu'][-1], self.states['servo'][-1], self.states['lidar'][-1]])
            imu_t, servo_t, lidar_t = self._obs_to_tensors(last_obs)
            with torch.no_grad():
                next_value = self.critic.get_value(imu_t, servo_t, lidar_t).item()

        advantages = self.calcAdvantage(
            rewards=self.rewards,
            values=self.values,
            next_value=next_value,
            dones=self.dones,
            gae_parameter=self.advantage  # advantage = lambda
        )

        returns = self.calcDiscountedReturns(rewards=self.rewards, dones=self.dones)

        # Convert data to tensor
        imu_tensor = torch.FloatTensor(np.array(self.states['imu']))
        servo_tensor = torch.FloatTensor(np.array(self.states['servo']))
        lidar_tensor = torch.FloatTensor(np.array(self.states['lidar']))
        actions_tensor = torch.FloatTensor(np.array(self.actions))
        old_log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).squeeze()

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
                
                # Update actor
                self.updateActor(batch_imu, batch_servo, batch_lidar, batch_actions, batch_old_log_probs, batch_advantages)
                
                # Update critic
                self.updateCritic(batch_imu, batch_servo, batch_lidar, batch_returns)

        self.clear_memory()
    

    def updateActor(self, imu, servo, lidar, actions, log_prob_old, advantages):
        """ Update policy using clipped PPO objective"""
        # with torch.GradientTape() as tape:
        log_prob_new = self.compute_log_prob(imu, servo, lidar, actions)
        
        # Calculate probability ratio = pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        ratio = torch.exp(log_prob_new - log_prob_old)
    
        # Calculate clipped vs unclipped loss
        unclippedL = ratio * advantages
        ratio_clipped  = torch.clamp(ratio, 1 - self.clipping, 1 + self.clipping)
        clippedL = ratio_clipped  * advantages

        # actor loss = pessimistic estimate, take minimum
        actor_loss = (-torch.min(unclippedL, clippedL)).mean()

        # Calculate gradients, backward propagation for actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    

    def updateCritic(self, imu, servo, lidar, returns):
        """
        From jennyf12-patch-1
    
        Train the value network with actual returns and update the critic

        Args:
              states: State tensor of shape (batch_size, state_dim)
              returns: Target returns of shape (batch_size,)

        Returns:
            avg_loss: Average loss across epochs
        """
        # Ensure returns has the correct shape
        if returns.dim() == 1:
            returns = returns.unsqueeze(-1)

        # Set critic to training mode
        self.critic.train()
        total_loss = 0.0

        # Train for multiple epochs
        for _ in range(self.epochs):
            # Forward Pass: predict state values
            predicted_values = self.critic(imu, servo, lidar)

            # Compute MSE loss: mean squared error between prediction and target
            loss = nn.MSELoss()(predicted_values, returns)

            # Back propagation
            # Clear previous gradients
            self.critic_optimizer.zero_grad()

            # Compute gradients of the loss
            loss.backward()

            # Gradient clipping, helps prevent exploding gradients
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            # Update network weights
            self.critic_optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Calculate average loss across all epochs
        avg_loss = total_loss / self.epochs
        return avg_loss
    

    def clear_memory(self):
        """
        from zihanwang126-branch
        Clear trajectory buffers
        Resets all storage lists to empty
        """
        self.states = {'imu': [], 'servo': [], 'lidar': []}
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        print("Buffers cleared")
