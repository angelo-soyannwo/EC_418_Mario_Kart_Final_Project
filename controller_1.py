import pystk

import numpy as np

from utils import PyTux



def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25, concerning_object_in_frame=None):


    # Initialize an action object

    action = pystk.Action()
    if concerning_object_in_frame != None:
        if (concerning_object_in_frame == 0 or  concerning_object_in_frame==2):

            # Turn sharply towards the aim point to avoid the object
            action.steer = np.clip(steer_gain * aim_point[0], -1, 1)

            # Apply heavy braking to slow down quickly
            action.acceleration = 0.0
            action.brake = True

            # If steer is extreme, enable drift
            if abs(action.steer) > 0.7:
                action.drift = True
            else:
                action.drift = False

            # Disable nitro for careful movement
            action.nitro = False
        else: #no concerning image
            # Calculate steering based on aim_point
            action.steer = np.clip(steer_gain * aim_point[0], -1, 1)
            velocity_diff = target_vel - current_vel

            if velocity_diff > skid_thresh:  # Speed up
                action.acceleration = 1.0
                action.brake = False
            elif velocity_diff < -skid_thresh:  # Slow down
                action.acceleration = 0.0
                action.brake = True
            else:  # Maintain speed
                action.acceleration = 0.3  # Minimal acceleration to maintain speed
                action.brake = False


            # Skid recovery
            action.nitro = abs(action.steer) < 0.1 and velocity_diff > 0  # Use nitro when going straight


    else:
        # Calculate steering based on aim_point

        action.steer = np.clip(steer_gain * aim_point[0], -1, 1)


        """
        # Adjust acceleration and brake based on current velocity and target velocity

        if current_vel < target_vel:

            action.acceleration = 1.0

            action.brake = False

        else:

            action.acceleration = 0.0

            action.brake = True if current_vel > target_vel + skid_thresh else False

        """
        velocity_diff = target_vel - current_vel

        if velocity_diff > skid_thresh:  # Speed up
            action.acceleration = 1.0
            action.brake = False
        elif velocity_diff < -skid_thresh:  # Slow down
            action.acceleration = 0.0
            action.brake = True
        else:  # Maintain speed
            action.acceleration = 0.3  # Minimal acceleration to maintain speed
            action.brake = False


        # Skid recovery
        action.nitro = abs(action.steer) < 0.1 and velocity_diff > 0  # Use nitro when going straight


    
    
    return action


def test_controller(pytux, track, verbose=False):

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
