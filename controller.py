import pystk
import numpy as np
from utils import PyTux
import random

# PID parameters for steering and velocity control
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.integral = 0
        self.prev_error = 0

    def compute(self, error, dt):
        self.integral += error * dt  # Integral term
        derivative = (error - self.prev_error) / dt  # Derivative term
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative  # PID output


# Instantiate PID controllers for steering and velocity
steer_pid = PIDController(kp=6, ki=0.1, kd=1.0)
velocity_pid = PIDController(kp=1.0, ki=0.1, kd=0.5)


def control(aim_point, current_vel, dt=0.1, target_vel=25, concerning_object_in_frame=None):
    """
    Control function with PID control for steering and velocity.
    """
    action = pystk.Action()

    # PID for steering
    steer_error = aim_point[0]  # Error is the horizontal distance from the target point
    action.steer = np.clip(steer_pid.compute(steer_error, dt), -1, 1)

    if concerning_object_in_frame is not None:
        if concerning_object_in_frame == 0 or concerning_object_in_frame == 2:
            # Slow down and turn sharply to avoid obstacles
            velocity_error = target_vel - current_vel
            acceleration = velocity_pid.compute(velocity_error, dt)
            action.acceleration = np.clip(acceleration, 0, 1.0)

            #if random.random() < 0.4 :
            #    action.brake = True
            action.drift = abs(action.steer) > 0.8 #was 0.7
        else:
            # PID for velocity
            velocity_error = target_vel - current_vel
            acceleration = velocity_pid.compute(velocity_error, dt)
            if acceleration > 0:
                action.acceleration = np.clip(acceleration, 0, 1.0)
                action.brake = False
            else:
                action.acceleration = 0.0
                action.brake = True
            action.drift = False
    else:
        # PID for velocity
        velocity_error = target_vel - current_vel
        acceleration = velocity_pid.compute(velocity_error, dt)
        if acceleration > 0:
            action.acceleration = np.clip(acceleration, 0, 1.0)
            action.brake = False
        else:
            action.acceleration = 0.0
            action.brake = True

        action.nitro = abs(action.steer) < 0.1 and acceleration > 0  # Use nitro when going straight

    return action


def test_controller(pytux, track, verbose=False):
    """
    Test the controller on specified tracks.
    """
    # Set track to list if it's a single string
    track = [track] if isinstance(track, str) else track

    # Loop through the tracks
    for t in track:
        # Rollout for 5 seconds (around 300 frames)
        steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=verbose)
        print(f'Steps: {steps}, Distance traveled: {how_far}')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')

    pytux = PyTux()
    test_controller(pytux, **vars(parser.parse_args()))
    pytux.close()

