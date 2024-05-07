#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2
import datetime
import json
import tf

class LaserScanProcessor:
    def __init__(self):
        self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
        self.odom_subscriber = rospy.Subscriber("/tracked_pose", PoseStamped, self.odom_callback)
        self.scans_received = 0
        self.scan_data = []
        self.last_odom = None
        self.last_odom_stamp = rospy.Time(0)

    def odom_callback(self, data):
        self.last_odom = data
        self.last_odom_stamp = data.header.stamp

    def scan_callback(self, data):
        # Only process scans if a recent odom message was received
        if self.last_odom_stamp.to_sec() > 0 and abs((data.header.stamp - self.last_odom_stamp).to_sec()) < 0.1:
            self.scans_received += 1
            if self.scans_received <= 1:
                self.scan_data.append(data)
                self.scan_data.append(data)
                self.scan_data.append(data)
                if self.scans_received == 1:
                    self.generate_images_and_save_pose(data.header.stamp)
                    self.scans_received = 0
                    self.scan_data = []

    # Function to check if a position is in the parking area
    def in_parking_area(self,x, y):
        return 0 <= x <= 9 and -9 <= y <= 9

    def generate_images_and_save_pose(self, scan_timestamp):
        # Initialize the images with a resolution of 448x448
        image_size = 448
        rgb_images = [np.full((image_size, image_size, 3), 255, dtype=np.uint8) for _ in range(3)]
        occupancy_grids = [np.full((image_size, image_size, 3), 128, dtype=np.uint8) for _ in range(3)]  # Start with unknown (grey)

        for idx, scan in enumerate(self.scan_data):
            angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
            robot_x, robot_y = image_size // 2, image_size // 2  # Robot position in image coordinates

            for angle, distance in zip(angles, scan.ranges):
                # Adjust distance for out-of-range measurements to treat as free space
                if distance == 0 or distance > scan.range_max:
                    distance = scan.range_max

                # Transform laser scan points to image coordinates
                x = int(robot_x + (distance * np.cos(angle)) * image_size / (scan.range_max * 2))
                y = int(robot_y + (distance * np.sin(angle)) * image_size / (scan.range_max * 2))

                # Draw a line for free space
                cv2.line(occupancy_grids[idx], (robot_x, robot_y), (x, y), (255, 255, 255), 1)

                # Check if the point is within the image bounds to mark as an obstacle
                if 0 <= x < image_size and 0 <= y < image_size:
                    if distance < scan.range_max:  # Only mark as an obstacle if not out-of-range
                        rgb_images[idx][y, x] = [0, 0, 0]
                        cv2.circle(occupancy_grids[idx], (x, y), 2, (0, 0, 0), -1)


        # Convert ROS Time to datetime object and then to a string for the timestamp
        timestamp_str = datetime.datetime.fromtimestamp(scan_timestamp.to_sec()).strftime("%Y%m%d-%H%M%S")
        # Combine the images to create a 6-channel image
        multi_channel_image = np.concatenate((*rgb_images, *occupancy_grids), axis=2)


        # Save the multi-channel image as a NumPy array

        np.save(f"/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data1scan/multi_channel_images/{timestamp_str}.npy", multi_channel_image)


        # # Save or display images
        cv2.imwrite(f"/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_newOccupancy/rgb_images/{timestamp_str}.jpg", rgb_images[0])
        cv2.imwrite(f"/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_newOccupancy/occupancy_grids/{timestamp_str}.jpg", occupancy_grids[1])

        # Save pose data with the same timestamp if it's recent enough
        if self.last_odom and abs((scan_timestamp - self.last_odom.header.stamp).to_sec()) < 0.1:
            orientation_q = self.last_odom.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

            pose_data = {
                'timestamp': scan_timestamp.to_sec(),  # Use the ROS Time object directly here
                'position': {
                    'x': self.last_odom.pose.position.x,
                    'y': self.last_odom.pose.position.y,
                },
                'orientation': yaw,  # Radians
                'parking': self.in_parking_area(self.last_odom.pose.position.x, self.last_odom.pose.position.y)
            }

            with open(f"/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data1scan/gt_poses/{timestamp_str}.json", 'w') as f:
                json.dump(pose_data, f)

if __name__ == '__main__':
    rospy.init_node('laser_scan_processor', anonymous=True)
    lsp = LaserScanProcessor()

    rospy.spin()
    

