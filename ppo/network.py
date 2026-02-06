
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate
from config import ACTION_SPACE_SIZE
import numpy as np

#Actor state input consists of IMU (6), Servo (12) if Lidar is added (3)

# COPIED FROM REFERENCE CODE - Most should be usable but some stuff should be changed for PPO
class Actor:
    def actor_model(self):
        imu_input = Input(shape =(6,), name="IMU_input") 
        x_imu = Dense(64, activation='relu')(imu_input)
        x_imu = Dense(32, activation='relu')(x_imu)
       
        servo_input = Input(shape=(12,), name="Servo_input")
        x_servo = Dense(64, activation='relu')(servo_input)
        x_servo = Dense(32, activation='relu')(x_servo)

        lidar_input = Input(shape=(3,), name="Lidar_input")
        x_lidar = Dense(64, activation='relu')(lidar_input)
        x_lidar = Dense(32, activation='relu')(x_lidar)

        combined_inputs = Concatenate()([x_imu, x_servo, x_lidar])
        x = Dense(64, activation='relu')(combined_inputs)
        
        # Output should be changed
        output = Dense(ACTION_SPACE_SIZE, activation='tanh')(x)
        
        actor_model = Model(inputs=[imu_input, servo_input, lidar_input], outputs=output)

        return actor_model

class Critic:
    #state and action inputs
    def critic_model(self):
        imu_input = Input(shape =(6,), name="IMU_input") 
        x_imu = Dense(64, activation='relu')(imu_input)
        x_imu = Dense(32, activation='relu')(x_imu)
       
        servo_input = Input(shape=(12,), name="Servo_input")
        x_servo = Dense(64, activation='relu')(servo_input)
        x_servo = Dense(32, activation='relu')(x_servo)

        lidar_input = Input(shape=(3,), name="Lidar_input")
        x_lidar = Dense(64, activation='relu')(lidar_input)
        x_lidar = Dense(32, activation='relu')(x_lidar)

        # action_input = Input(shape=(ACTION_SPACE_SIZE,), name="Action_input")
        combined_inputs = Concatenate()([x_imu, x_servo, x_lidar])
        x = Dense(64, activation='relu')(combined_inputs)

        #q value output
        output = Dense(1, activation='linear')(x)

        critic_model = Model(inputs=[imu_input, servo_input, lidar_input], outputs=output)

        return critic_model
