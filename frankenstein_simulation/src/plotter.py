#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Path
import matplotlib.pyplot as plt
import os
import rospkg

class PathPlotter:
    def __init__(self):
        rospack = rospkg.RosPack()
        self.package_path = rospack.get_path('frankenstein_simulation')
        rospy.init_node('path_plotter', anonymous=True)

        # Subscribers
        self.odom_sub = rospy.Subscriber("/odom_path", Path, self.odom_callback)
        self.gt_sub = rospy.Subscriber("/gt_path", Path, self.gt_callback)
        self.slam_sub = rospy.Subscriber("/slam_path", Path, self.slam_callback)

        # Path data
        self.odom_path = []
        self.gt_path = []
        self.slam_path = []

    def odom_callback(self, data):
        self.path_callback(data, self.odom_path)

    def gt_callback(self, data):
        self.path_callback(data, self.gt_path)

    def slam_callback(self, data):
        # For the SLAM path, which is a PoseStamped instead of Odometry
        self.path_callback(data, self.slam_path)

    def path_callback(self, data, path_list):
        for pose in data.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            # More robust check could be added here based on your criteria
            path_list.append((x, y))

    def plot_paths(self):
        
        plt.figure(figsize=(10, 6))
        if self.odom_path:
            odom_x, odom_y = zip(*self.odom_path)
            plt.plot(odom_x, odom_y, 'ro', label='Odometry')  # Red line
        if self.gt_path:
            gt_x, gt_y = zip(*self.gt_path)
            plt.plot(gt_x, gt_y, 'g^', label='Ground Truth')  # Green line
        if self.slam_path:
            slam_x, slam_y = zip(*self.slam_path)
            plt.plot(slam_x, slam_y, 'bs', label='SLAM')  # Blue line
        # Setting plot parameters
        plt.xlabel('X position')
        plt.ylabel('Y position')
        plt.title('Path Comparison')
        plt.legend()

        # Ensure the data directory exists
        data_dir = os.path.join(self.package_path, 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Save the figure
        figure_path = os.path.join(data_dir, 'path_comparison.png')
        plt.savefig(figure_path)
        rospy.loginfo('Plot saved to {}'.format(figure_path))
        plt.close() 

    def run(self):
        rospy.on_shutdown(self.plot_paths)  # Save plot on shutdown
        rospy.spin()

if __name__ == '__main__':
    plotter = PathPlotter()
    plotter.run()
