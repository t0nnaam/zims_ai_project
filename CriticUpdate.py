import torch
import torch.nn as nn
import torch.optim as optim


class CriticUpdater:
    """
    CriticUpdater - Trains the value function to estimate state values
    """

    def __init__(self, critic, lr=3e-4, epochs=10, max_grad_norm=0.5):
        """
        Initialize the CriticUpdater

        Args:
              critic: Value network
              lr: Learning rate for optimizer
              epochs: Number of training epochs per update
              max_grad_norm: Maximum norm for gradient clipping
        """
        self.critic = critic
        self.optimizer = optim.Adam(critic.parameters(), lr=lr)
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm

    def update_critic(self, states, returns):
        """
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

