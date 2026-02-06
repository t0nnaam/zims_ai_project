# COPIED FROM REFERENCE CODE

from robot import Robot

MOVE_REWARD = 1
FALL_PENALTY = 1

#needs to be defined

#Max values determined the threshold of how much the robot can tilt before falling
#MAX_ROLL_VAL
#MAX_PITCH_VAL

#MAX_STEP_COUNT
class RobotEnv:       
    def reset(self):
        self.robot = Robot() #robot is already intialized to be at default position
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        return self.robot.get_state()
    
    #need to make sure action is vector of continuous values
    def step(self, action):
        reward = 0
        done = False
        self.robot.move(action)
        self.step_count += 1

        pitch, roll, yaw = self.robot.orientation

        if abs(roll) > MAX_ROLL_VAL or abs(pitch) > MAX_PITCH_VAL:
            reward -= FALL_PENALTY
            done = True
        elif self.step_count >= MAX_STEP_COUNT:
            done = True
        else:
            reward += MOVE_REWARD

        return self.get_state(), reward, done



    