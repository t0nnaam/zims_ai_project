import numpy as np
import torch


class TrajectoryCollector:
    """
    Person 1: Trajectory Collector
    Responsible for running the agent in the environment and collecting training data
    """

    def __init__(self, actor, critic):
        """
        Initialize the trajectory collector

        Args:
            actor: Actor network (decides actions)
            critic: Critic network (evaluates states)
        """
        self.actor = actor
        self.critic = critic

        # Buffers to store trajectory data
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def rollout(self, env, num_steps):
        """
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

            # 1. Convert state to tensor format
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # 2.  use Actor to sample action (actor's distribution)
            with torch.no_grad():
                action, log_prob = self.actor.get_action(state_tensor)

            # 3. Get value estimate from critic
            with torch.no_grad():
                value = self.critic.get_value(state_tensor)

            # 4. Convert action to numpy array (environment expects numpy)（tensor → numpy）
            if isinstance(action, torch.Tensor):
                action_np = action.cpu().numpy().flatten()
            else:
                action_np = action

            # 5. Execute action in environment
            next_state, reward, done, info = env.step(action_np)

            # 6. Store transition data
            self.states.append(state)
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
        Get collected data and convert to numpy arrays

        Returns:
            Dictionary containing all trajectory data
        """
        data = {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values).flatten(),
            'dones': np.array(self.dones)
        }
        return data

    def clear_memory(self):
        """
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