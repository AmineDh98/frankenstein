#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
import numpy as np
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion

class OdometryErrorCalculator:
    def __init__(self):
        # Initialize node
        rospy.init_node('odometry_error_calculator')

        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.gt_sub = rospy.Subscriber('/ground_truth', Odometry, self.gt_callback)

        # Data storage
        self.odom_data = []
        self.gt_data = []
        self.times = []

        # Start time
        self.start_time = rospy.Time.now()

    def odom_callback(self, msg):
        if rospy.is_shutdown():
            return
        self.odom_data.append((msg.pose.pose.position.x, msg.pose.pose.position.y, self.quaternion_to_yaw(msg.pose.pose.orientation)))

    def gt_callback(self, msg):
        if rospy.is_shutdown():
            return
        time_elapsed = (msg.header.stamp - self.start_time).to_sec()
        self.gt_data.append((msg.pose.pose.position.x, msg.pose.pose.position.y, self.quaternion_to_yaw(msg.pose.pose.orientation)))
        self.times.append(time_elapsed)
        if self.odom_data:
            self.calculate_and_plot_errors()

    def quaternion_to_yaw(self, orientation):
        _, _, yaw = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
        return yaw

    def calculate_and_plot_errors(self):
        if len(self.odom_data) != len(self.gt_data) or len(self.odom_data) == 0:
            return

        # Convert to numpy arrays for easier manipulation
        odom_array = np.array(self.odom_data)
        gt_array = np.array(self.gt_data)

        # Calculate errors
        position_errors = np.linalg.norm(odom_array[:, :2] - gt_array[:, :2], axis=1)
        angular_errors = np.abs(odom_array[:, 2] - gt_array[:, 2])

        # Plot results
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.times, position_errors, label='Position Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (m)')
        plt.title('Position Error over Time')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.times, angular_errors, label='Angular Error')
        plt.xlabel('Time (s)')
        plt.ylabel('Error (rad)')
        plt.title('Angular Error over Time')
        plt.legend()

        plt.tight_layout()
        plt.savefig('odometry_error_plot.png')
        # plt.show()

        # # Save errors to file
        # np.savetxt('odometry_position_errors.csv', position_errors, delimiter=',')
        # np.savetxt('odometry_angular_errors.csv', angular_errors, delimiter=',')

if __name__ == '__main__':
    calculator = OdometryErrorCalculator()
    while not rospy.is_shutdown():
        rospy.spin()
