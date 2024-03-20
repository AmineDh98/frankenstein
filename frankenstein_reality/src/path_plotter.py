#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Global lists to store paths
odom_path_x, odom_path_y = [], []
gt_path_x, gt_path_y = [], []
slam_path_x, slam_path_y = [], []

# Callback functions for each subscriber
def odom_path_callback(msg):
    global odom_path_x, odom_path_y
    odom_path_x = [pose.pose.position.x for pose in msg.poses]
    odom_path_y = [pose.pose.position.y for pose in msg.poses]

def gt_path_callback(msg):
    global gt_path_x, gt_path_y
    gt_path_x = [pose.pose.position.x for pose in msg.poses]
    gt_path_y = [pose.pose.position.y for pose in msg.poses]

def slam_path_callback(msg):
    global slam_path_x, slam_path_y
    slam_path_x = [pose.pose.position.x for pose in msg.poses]
    slam_path_y = [pose.pose.position.y for pose in msg.poses]

# Function to update the plot
def update_plot(frame):
    plt.cla()  # Clear the current axes
    # Plot Odom Path
    if odom_path_x and odom_path_y:
        min_length = min(len(odom_path_x), len(odom_path_y))
        plt.plot(odom_path_x[:min_length], odom_path_y[:min_length], 'r-', label='Odom Path')
    # Plot GT Path
    if gt_path_x and gt_path_y:
        min_length = min(len(gt_path_x), len(gt_path_y))
        plt.plot(gt_path_x[:min_length], gt_path_y[:min_length], 'g--', label='GT Path')
    # Plot SLAM Path
    if slam_path_x and slam_path_y:
        min_length = min(len(slam_path_x), len(slam_path_y))
        plt.plot(slam_path_x[:min_length], slam_path_y[:min_length], 'b-.', label='SLAM Path')
    plt.legend(loc='best')
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Paths Comparison')
    plt.axis('equal')  # Set equal scaling (i.e., 1 unit in x equals 1 unit in y)
    plt.grid(True)  # Show grid


if __name__ == '__main__':
    rospy.init_node('path_plotter', anonymous=True)

    # Subscribers
    rospy.Subscriber('odom_path', Path, odom_path_callback)
    rospy.Subscriber('gt_path', Path, gt_path_callback)
    rospy.Subscriber('slam_path', Path, slam_path_callback)

    # Setup plot
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, update_plot, interval=1000)

    plt.show(block=True)  # This will block the script until the plot window is closed

    rospy.spin()
