#!/usr/bin/env python3

import rospy
import numpy as np
import matplotlib.pyplot as plt
import message_filters
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from tf.transformations import euler_from_quaternion
import math

class OdometryErrorCalculator:
    def __init__(self):
        rospy.init_node('odometry_error_calculator')

        # Subscribers
        self.odom_sub = message_filters.Subscriber('/odom', Odometry)
        self.gt_sub = message_filters.Subscriber('/ground_truth', Odometry)
        self.joint_state_sub = message_filters.Subscriber('/frankenstein/joint_states', JointState)

        # Synchronize the topics
        ts = message_filters.ApproximateTimeSynchronizer([self.odom_sub, self.gt_sub, self.joint_state_sub], 10, 0.1, allow_headerless=True)
        ts.registerCallback(self.callback)

        # Data storage for plotting
        self.last_odom_pos = None
        self.last_gt_pos = None
        self.times = []
        self.displacement_errors = []
        self.angular_errors = []
        self.velocities = []
        self.steering_angles = []

        self.x_displacements = []
        self.y_displacements = []

    def callback(self, odom_msg, gt_msg, joint_msg):
        current_time = odom_msg.header.stamp.to_sec()
        self.times.append(current_time)
        
        # Extract position and orientation data
        current_odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        current_gt_pos = np.array([gt_msg.pose.pose.position.x, gt_msg.pose.pose.position.y])
        current_odom_yaw = self.quaternion_to_yaw(odom_msg.pose.pose.orientation)
        current_gt_yaw = self.quaternion_to_yaw(gt_msg.pose.pose.orientation)

        # Extract velocity and steering angle
        velocity = joint_msg.position[0] if joint_msg.position else 0
        velocity = velocity/1000
        steering_angle = joint_msg.position[1]
        steering_angle = self.deg_to_rad(steering_angle)
        steering_angle = self.normalize_angle_rad(steering_angle)

        if self.last_odom_pos is not None and self.last_gt_pos is not None:
            # Calculate displacements and errors
            odom_displacement = np.linalg.norm(current_odom_pos - self.last_odom_pos)
            gt_displacement = np.linalg.norm(current_gt_pos - self.last_gt_pos)
            displacement_error = abs(odom_displacement - gt_displacement)
            
            x_displacement_odom = current_odom_pos[0] - self.last_odom_pos[0]
            y_displacement_odom = current_odom_pos[1] - self.last_odom_pos[1]

            x_displacement_gt = current_gt_pos[0] - self.last_gt_pos[0]
            y_displacement_gt = current_gt_pos[1] - self.last_gt_pos[1]


            x_displacement = abs(x_displacement_odom - x_displacement_gt)
            y_displacement = abs(y_displacement_odom - y_displacement_gt)

            angular_error = abs(current_odom_yaw - self.last_odom_yaw) - abs(current_gt_yaw - self.last_gt_yaw)
            angular_error = self.normalize_angle_rad(angular_error)
            angular_error = abs(angular_error)  # Ensure the error is non-negative

            # Store data
            self.displacement_errors.append(displacement_error)
            self.angular_errors.append(angular_error)
            self.velocities.append(velocity)
            self.steering_angles.append(steering_angle)
            self.x_displacements.append(x_displacement)
            self.y_displacements.append(y_displacement)

        # Update last known positions and orientations
        self.last_odom_pos = current_odom_pos
        self.last_gt_pos = current_gt_pos
        self.last_odom_yaw = current_odom_yaw
        self.last_gt_yaw = current_gt_yaw

        # Plot every few messages
        if len(self.displacement_errors) % 10 == 0 and len(self.displacement_errors) !=0 :
            self.plot_errors()

    def quaternion_to_yaw(self, orientation):
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        return yaw
    def deg_to_rad(selfe, deg):
        return deg * math.pi / 180.0
    def normalize_angle_rad(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def plot_errors(self):
        plt.figure(figsize=(20, 10))
        x = len(self.displacement_errors)
        y = len(self.times)
        a = len(self.angular_errors)
        z = min(x,y)
        b = min(a,y)
        # Plot Position Error over Time
        plt.subplot(2, 3, 1)
        plt.plot(self.times[:z], self.displacement_errors[:z], label='Position Error (m)')
        plt.xlabel('Time (s)')
        plt.ylabel('Displacement Error (m)')
        plt.title('Position Displacement Error over Time')
        plt.legend()

        # Plot Angular Error over Time
        plt.subplot(2, 3, 2)
        plt.plot(self.times[:b], self.angular_errors[:b], label='Angular Error (rad)')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Error (rad)')
        plt.title('Angular Displacement Error over Time')
        plt.legend()

        # Plot Position Error vs Velocity
        plt.subplot(2, 3, 3)
        plt.scatter(self.velocities, self.displacement_errors, c='r', label='Displacement Error vs Velocity')
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Displacement Error (m)')
        plt.title('Displacement Error vs Velocity')
        plt.legend()

        # Plot Angular Error vs Steering Angle
        plt.subplot(2, 3, 4)
        plt.scatter(self.steering_angles, self.angular_errors, c='b', label='Angular Error vs Steering Angle')
        plt.xlabel('Steering Angle (rad)')
        plt.ylabel('Angular Error (rad)')
        plt.title('Angular Error vs Steering Angle')
        plt.legend()

        # X Displacement vs Steering Angle
        plt.subplot(2, 3, 5)
        plt.scatter(self.steering_angles, self.x_displacements, c='g', label='X Displacement error vs Steering Angle')
        plt.xlabel('Steering Angle (rad)')
        plt.ylabel('X Displacement error (m)')
        plt.title('X Displacement error vs Steering Angle')
        plt.legend()

        # Y Displacement vs Steering Angle
        plt.subplot(2, 3, 6)
        plt.scatter(self.steering_angles, self.y_displacements, c='m', label='Y Displacement error vs Steering Angle')
        plt.xlabel('Steering Angle (rad)')
        plt.ylabel('Y Displacement error (m)')
        plt.title('Y Displacement error vs Steering Angle')
        plt.legend()

        plt.tight_layout()
        plt.savefig('/home/aminedhemaied/catkin_ws/src/frankenstein/odometry_error_analysis.png')
        # plt.show()

if __name__ == '__main__':
    calculator = OdometryErrorCalculator()
    rospy.spin()
