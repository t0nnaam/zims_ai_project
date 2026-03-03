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
        self.actor_optimizer = torch.optim.Adam(learning_rate=0.001)
        self.critic_optimizer = torch.optim.Adam(learning_rate=0.001)
        
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

    def compute_log_prob(self, imu, servo, lidar, actions):
        pass

    def rollout(self):
        """ Get trajectory data by executing policy within environment """
        pass

    def calcAdvantage(self):
        """ Calculate Advantage Estimation """
        pass

    def calcDiscountedReturns(self):
        """ Calculate Discounted Returns """
        pass
    
    def update(self):
        """ Main update function - call actor and critic updates"""
        pass
    
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

            # Backpropagation
            #Clear previous gradients
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
    
    def clearMemory(self):
        """ Clear trajectory buffers after update"""
        pass

    # Should we add a function to log all the values?
