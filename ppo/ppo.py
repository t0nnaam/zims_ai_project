# from import *
from network import Actor, Critic
# from replay_buffer import ReplayBuffer
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam

# Make a branch when you work on the code and then push it to the repo

# I started an outline but we still need to change the parameters on the functions

class PPO:
    def __init__(self, discount, clipping, advantage, batch_size):
        # Networks
        self.actor = Actor()
        self.critic = Critic()
        
        # Optimizers
        self.actor_optimizer = Adam(learning_rate=0.001)
        self.critic_optimizer = Adam(learning_rate=0.001)
        
        # Hyperparameters
        self.discount = discount  # discount factor
        self.clipping = clipping  # PPO clipping parameter
        self.advantage = advantage # for advantage estimation
        self.batch_size = batch_size
        
        # Storage for trajectories
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []

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
    
    def update_actor(self):
        """ Update policy using clipped PPO objective"""
        pass
    
    def update_critic(self, states, returns):
        """ Update value function using MSE loss function"""
        pass
    
    def clear_memory(self):
        """ Clear trajectory buffers after update"""
        pass

    # Should we add a function to log all the values?
