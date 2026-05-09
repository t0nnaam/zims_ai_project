# COPIED FROM REFERENCE CODE

import numpy as np

#these need to be defined
#DEFAULT_SERVO_POSITIONS = 
#MOVE_FORWARD =
#TURN_LEFT =
#TURN_RIGHT =
#SERVO_MIN = 
#SERVO_MAX = 
#GAIT CYCLES need to be defined

class Robot:
    def __init__(self):
        self.servo_positions = DEFAULT_SERVO_POSITIONS
        self.orientation = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        
    def take_action(self, choice):
       pass
    
    #functions need to be further defined by gait pattern and orientation changes
    def update_servo_position(self):
        pass

    def update_orientation(self):
        pass

    def get_state(self):
        return np.concatenate([
            self.orientation,
            self.angular_velocity,
            self.servo_positions
        ])
        