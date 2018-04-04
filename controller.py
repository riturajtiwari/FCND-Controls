"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
import math
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0

class NonlinearController(object):

    def __init__(self):
        """Initialize the controller object and control gains"""
        self.z_k_p = 1.0
        self.z_k_d = 1.0
        self.x_k_p = 1.0
        self.x_k_d = 1.0
        self.y_k_p = 1.0
        self.y_k_d = 1.0
        self.k_p_roll = 1.0
        self.k_p_pitch = 1.0
        self.k_p_yaw = 1.0
        self.k_p_p = 1.0
        self.k_p_q = 1.0
        self.k_p_r = 1.0

        print('x: delta = %5.3f' % (self.x_k_d / 2 / math.sqrt(self.x_k_p)), ' omega_n = %5.3f' % (math.sqrt(self.x_k_p)))
        print('y: delta = %5.3f' % (self.y_k_d / 2 / math.sqrt(self.y_k_p)), ' omega_n = %5.3f' % (math.sqrt(self.y_k_p)))
        print('z: delta = %5.3f' % (self.z_k_d / 2 / math.sqrt(self.z_k_p)), ' omega_n = %5.3f' % (math.sqrt(self.z_k_p)))

        self.g = 9.81
        return    

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory
        
        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds
            
        Returns: tuple (commanded position, commanded velocity, commanded yaw)
                
        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]
        
        
        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]
            
            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]
            
        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]
                
                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]
            
        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)
        
        
        return (position_cmd, velocity_cmd, yaw_cmd)
    
    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command
            
        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        return np.array([0.0, 0.0])
    
    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude, acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            acceleration_ff: feedforward acceleration command (+up)
            
        Returns: thrust command for the vehicle (+up)
        """
        z_err = altitude_cmd - altitude
        z_err_dot = vertical_velocity_cmd - vertical_velocity
        rot_mat = self.attitude_to_rot_mat(attitude)
        b_z = rot_mat[2, 2]

        p_term = self.z_k_p * z_err
        d_term = self.z_k_d * z_err_dot

        u_1_bar = p_term + d_term + acceleration_ff

        c = (u_1_bar - self.g) / b_z

        return c
        
    
    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame
        
        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll,pitch,yaw) in radians
            thrust_cmd: vehicle thruts command in Newton
            
        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        return np.array([0.0, 0.0])
    
    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame
        
        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            attitude: 3-element numpy array (p,q,r) in radians/second^2
            
        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        (p_c, q_c, r_c) = body_rate_cmd
        (p_actual, q_actual, r_actual) = body_rate
        p_err = p_c - p_actual
        u_bar_p = self.k_p_p * p_err

        q_err = q_c - q_actual
        u_bar_q = self.k_p_q * q_err

        r_err = r_c - r_actual
        u_bar_r = self.k_p_r * r_err

        return u_bar_p, u_bar_q, u_bar_r
    
    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate
        
        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians
        
        Returns: target yawrate in radians/sec
        """
        psi_err = yaw_cmd - yaw
        r_c = self.k_p_yaw * psi_err

        return r_c

    @staticmethod
    def attitude_to_rot_mat(attitude):
        (phi, theta, psi) = attitude
        euler_rot_mat = np.array([[1, math.sin(phi) * math.tan(theta), math.cos(phi) * math.tan(theta)],
                                  [0, math.cos(phi), -math.sin(phi)],
                                  [0, math.sin(phi) / math.cos(theta), math.cos(phi) / math.cos(theta)]])
        return euler_rot_mat
