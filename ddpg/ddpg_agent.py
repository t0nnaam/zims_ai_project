from ddpg.config import *
from network import Actor, Critic
from replay_buffer import ReplayBuffer
import random
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam

class DDPGAgent:
    def __init__(self):
        self.actor_network = Actor()
        self.critic_network = Critic()

        self.target_actor_network = Actor()
        self.target_critic_network = Critic()

        self.target_actor_network.set_weights(self.actor_network.get_weights())
        self.target_critic_network.set_weights(self.critic_network.get_weights())

        self.replay_buffer = ReplayBuffer()

        self.actor_optimizer = Adam(learning_rate=0.001)
        self.critic_optimizer = Adam(learning_rate=0.001)

    def update_replay_buffer(self, transition):
        self.replay_buffer.add(transition)

    def choose_action(self, state):
        actor_model = self.actor_network.actor_model()
        action = actor_model.predict(state)[0]
        action = np.clip(action, -1, 1)
        return action
    
    def train(self):
        if len(self.replay_memory) < MIN_REPLAY_BUFFER_SIZE:
            return
        
        #minibatch returns (state, action, reward, next_state, done)
        minibatch = random.sample(self.replay_buffer, MINIBATCH_SIZE)
        
        current_imu_state = np.array([transition[0][0] for transition in minibatch])
        current_servo_state = np.array([transition[0][1] for transition in minibatch])
        current_states = [current_imu_state, current_servo_state]

        actor_actions = self.actor_network.actor_model().predict(current_states)
        actor_actions = np.clip(actor_actions, -1, 1)                               #actor_actions are continuos so clip

        new_imu_state = np.array([transition[3][0] for transition in minibatch])
        new_servo_state = np.array([transition[3][1] for transition in minibatch])
        new_current_states = [new_imu_state, new_servo_state]
        
        target_actor_actions = self.target_actor_network.actor_model().predict(new_current_states)
        target_actor_actions = np.clip(target_actor_actions, -1, 1)

        target_critic_input = [new_imu_state, new_servo_state, target_actor_actions]

        target_critic_q_value = self.target_critic_network.critic_model().predict(target_critic_input)

        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch]).reshape(-1, 1)
        dones = np.array([transition[4] for transition in minibatch]).reshape(-1, 1)

        #Bellmanford
        y = rewards + DISCOUNT * (1 - dones) * target_critic_q_value

        #train critic 
        self.critic_network.critic_model().train_on_batch(
            [current_imu_state, current_servo_state, actions],
            y
        )

        #train actor using optimizer
        with tf.GradientTape() as tape:
            actor_actions = self.actor_network.actor_model()(current_states, training=True)
            critic_q_values = self.critic_network.critic_model()(
                [current_imu_state, current_servo_state, actor_actions],
                training=True
            )
            actor_loss = -tf.reduce_mean(critic_q_values)

        actor_gradients = tape.gradient(
            actor_loss,
            self.actor_network.actor_model().trainable_variables
        )
        
        self.actor_network.optimizer.apply_gradients(
            zip(actor_gradients, self.actor_network.actor_model().trainable_variables)
        )

        #update target networks
        self.update_target_network(
            self.target_actor_network.actor_model(),
            self.actor_network.actor_model(),
            TAU
        )

        self.update_target_network(
            self.target_critic_network.critic_model(),
            self.critic_network.critic_model(),
            TAU
        )

            

