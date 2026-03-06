# from import *
from network import Actor, Critic
# from replay_buffer import ReplayBuffer
import random
import torch
import torch.nn as nn
import numpy as np
# from tensorflow.keras.optimizers import Adam

# Make a branch when you work on the code and then push it to the repo
 
# I started an outline but we still need to change the parameters on the functions

class PPO:
    def __init__(self, discount=0.99, clipping=0.2, advantage=0.9, epoch=10, batch_size=64):
        # the values here are sorta placeholders rn - we can test and change later
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
        
        # Storage for trajectories
        self.states = {'imu': [], 'servo': [], 'lidar':[]}
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = [] # store old log probs for PPO
        self.dones = [] # added from zihanwang126-branch
        # self.rewards_tensor = torch.tensor(self.rewards, dtype = torch.float32)
        # self.critic_values_tensor = self.critic.getValues(self.states['imu'], self.states['servo'], self.states['lidar']).detach()

    def compute_log_prob(self, imu, servo, lidar, actions):
        pass

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

        
        # Collect num_steps of data
        for step in range(num_steps):
            #split each sensor into seperate paramaters for get_action and get_value
            imu = state[:6]
            servo = state[6:18]
            lidar = state[18:21]
            #make each one into a tensor
            imu_tensor = torch.FloatTensor(imu).unsqueeze(0)
            servo_tensor = torch.FloatTensor(servo).unsqueeze(0)
            lidar_tensor = torch.FloatTensor(lidar).unsqueeze(0)

            # 1. Convert state to tensor format
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 2.  use Actor to sample action (actor's distribution)
            with torch.no_grad():
                action, log_prob = self.actor.get_action(imu_tensor, servo_tensor, lidar_tensor)

            # 3. Get value estimate from critic
            with torch.no_grad():
                value = self.critic.get_value(imu_tensor, servo_tensor, lidar_tensor)

            # 4. Convert action to numpy array (environment expects numpy)（tensor → numpy）
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy().flatten()
            else:
                action_np = action

            # 5. Execute action in environment
            next_state, reward, done, info = env.step(action_np)

            # 6. Store transition data
            imu = state[:6]
            servo = state[6:18]
            lidar = state[18:21]
            
            self.states['imu'].append(imu)
            self.states['servo'].append(servo)
            self.states['lidar'].append(lidar)
            self.actions.append(action_np)
            self.rewards.append(reward)
            self.log_probs.append(log_prob.cpu().numpy())
            self.values.append(value.cpu().numpy())
            self.dones.append(done)

            # 7. Update state and statistics
            state = next_state
            episode_reward += reward

            # 8. Reset environment if episode is done
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

    def calculateTDResidual(self, next_value, done): 
        mask_tensor = 1.0 - torch.tensor(done, dtype = torch.float32)

        next_values = torch.cat([self.critic_values_tensor[1:], next_value.unsqueeze(0)]) 
        # uses vectorization to calculate the TD residual 
        # current reward + (discount factor * next critic value) - current critic value 
        return self.rewards_tensor + (self.discount * mask_tensor * next_values) - self.critic_values_tensor[:-1]

    def calcAdvantage(self, next_value, next_advantage, done = False, gae_parameter = 0.95):
        """ Calculate Advantage Estimation """
        if done: 
            next_advantage = 0 
        return self.calculateTDResidual(next_value) + (self.discount * gae_parameter * next_advantage)

    def calcDiscountedReturns(self):
        """ Calculate Discounted Returns """
	    # (discounted factor ^ range from 0 to the number of rows of the rewards tensor) * the rewards tensor
        return self.discount ** torch.arange(self.rewards_tensor.size(0)) * self.rewards_tensor

    def update(self):
        """ Main update function - call actor and critic updates """
        # TODO : Get next_value
        next_value = 0.0

        # advantage_new = self.calcAdvantage(self) 
        advantage_new = self.calcAdvantage(
            rewards=self.rewards,
            values=self.values,
            next_value=next_value,
            dones=self.dones,
            gae_parameter=self.advantage  # using your 'advantage' hyperparameter as lambda
        )

        # discountedReturns_new = self.calcDiscountedReturns(self) 
        discountedReturns_new = self.calcDiscountedReturns(
            rewards=self.rewards,
            dones=self.dones
        )
        
        # May need to normalize advantages
        # TODO: Convert data to tensor

        # TODO: Update for multiple epochs
        # TODO: Create Minibatches (helper function)
        
        # self.updateActor(self, imu, servo, lidar, actions, log_prob_old) 
        self.updateActor(self, self.states['imu'], self.states['servo'], self.states['lidar'], self.actions, self.log_probs) 
        
        # self.updateCritic(self, states, returns) 
        self.updateCritic(self, self.states, self.returns) 

        self.clear_memory()
    
    def updateActor(self, imu, servo, lidar, actions, log_prob_old):
        """ Update policy using clipped PPO objective"""
        # with torch.GradientTape() as tape:
        log_prob_new = self.compute_log_prob(imu, servo, lidar, actions)
        
        # Calculate probability ratio = pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        ratio = torch.exp(log_prob_new - log_prob_old)
    
        # Calculate clipped vs unclipped loss
        unclippedL = ratio * self.advantage 
        ratio_clipped  = torch.clamp(ratio, 1 - self.clipping, 1 + self.clipping)
        clippedL = ratio_clipped  * self.advantage

        # actor loss = pessimistic estimate, take minimum
        actor_loss = (-torch.min(unclippedL, clippedL)).mean()

        # Calculate gradients, backward propagation for actor network
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def updateCritic(self, states, returns):
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
            predicted_values = self.critic(states)

            # Compute MSE loss: mean squared error between prediction and target
            loss = nn.MSELoss()(predicted_values, returns)

            # Back propagation
            # Clear previous gradients
            self.optimizer.zero_grad()

            # Compute gradients of the loss
            loss.backward()

            # Gradient clipping, helps prevent exploding gradients
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

            # Update network weights
            self.optimizer.step()

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
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
        print("Buffers cleared")

    # Should we add a function to log all the values?
