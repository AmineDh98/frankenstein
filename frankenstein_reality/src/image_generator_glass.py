#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import sensor_msgs.point_cloud2 as pc2
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
        self.old_scans_received = 0
        self.scan_data = []
        self.old_scans = []

        self.last_odom = None
        self.last_odom_stamp = rospy.Time(0)

        self.window_size = 9
        self.threshold = 6 #6
        self.image_size = 448
        self.max_range = 50.0 
        self.intensity_min = 0
        self.intensity_max = 10000
        self.intens = 4500
        self.width = 35

        self.angle_min = 0.0
        self.angle_max = 0.0

    def odom_callback(self, data):
        self.last_odom = data
        self.last_odom_stamp = data.header.stamp
    def scan_callback(self, msg):
        self.process_scan(msg)
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
    def process_scan(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        intensities = np.array(scan_msg.intensities) if hasattr(scan_msg, 'intensities') else np.zeros_like(ranges)
        angles = np.linspace(scan_msg.angle_min, scan_msg.angle_max, len(ranges))
        
        glass_indices, non_glass_indices, updated_ranges = self.detect_glass(ranges, intensities)
        old_scan = scan_msg
        scan_msg.ranges = updated_ranges
        self.coordinate(scan_msg, old_scan, glass_indices)

    def coordinate(self,scan_msg, old_scan, glass_indices):
        # Only process scans if a recent odom message was received
        if self.last_odom_stamp.to_sec() > 0 and abs((scan_msg.header.stamp - self.last_odom_stamp).to_sec()) < 0.1:
            self.scans_received += 1
            self.old_scans_received += 1
            
            if self.scans_received <= 3:
                self.scan_data.append(scan_msg)
                self.old_scans.append(old_scan)
                if self.scans_received == 3:
                    
                    self.generate_images_and_save_pose(scan_msg.header.stamp, glass_indices)
                    self.scans_received = 0
                    self.old_scans_received = 0

                    self.scan_data = []
                    self.old_scans = []


 
    
 


    def detect_glass(self, ranges, intensities):
        thresh = 40000
        grad = 25000
        grad_min = 2000
        width = 500

        target_angles = np.array([np.pi/2, -np.pi/2, 0, np.pi])
        glass_indices = []
        non_glass_indices = []
        isGlass = np.zeros(len(intensities), dtype=int)  # Initialize as not glass

        if len(intensities) == 0:
            return glass_indices, non_glass_indices, ranges, intensities

        # Initial setup based on the first point's intensity
        h = 0 if intensities[0] < thresh else 1
        angle_increment = (self.angle_max - self.angle_min) / len(ranges)

        angle_h = self.angle_min + h * angle_increment
        angle_i = self.angle_min  # Initialize outside the loop

        for i, intensity in enumerate(intensities[1:], start=1):
            angle_i += angle_increment
            
            if 4500 > intensity >= 0.0 and intensities[i - 1] > 5000 and -grad_min >= (intensity - intensities[i - 1]) >= -grad:
                h = i-1
                angle_h = self.angle_min + h * angle_increment  # Update angle_h

            elif 25000 > intensity >= 5000 and intensities[i - 1] < 4500 and grad_min <= (intensity - intensities[i - 1]) < grad:
                D1 = ranges[h]
                D2 = ranges[i]
                N = i - h
                Uc = (D2 - D1) / N
                
                if 3 <= N <= width and abs(D1-D2) < 7 and Uc < 0.2:
                    midpoint_angle = (angle_h + angle_i) / 2
                    dir_angle_h = np.arctan2(np.sin(midpoint_angle - angle_h), np.cos(midpoint_angle - angle_h))

                    if np.abs(dir_angle_h) < np.pi / 10 and np.any(np.abs(midpoint_angle - target_angles) < np.pi / 10):  
                        isGlass[h:i+1] = 1  # Update isGlass directly for the range of indices
                        ranges[h:i] = D1 + Uc * np.arange(N)  # Vectorized update of ranges
                        intensities[h:i+1] = 12000  # Vectorized update of intensities
                        glass_indices.extend(range(h, i+1))

        non_glass_indices = np.where(isGlass == 0)[0]
        return glass_indices, non_glass_indices, ranges
    
    
   


    def generate_images_and_save_pose(self, scan_timestamp, glass_indices):


        # Initialize the images with a resolution of 448x448
        image_size = 448
        rgb_images = [np.full((image_size, image_size, 3), 255, dtype=np.uint8) for _ in range(3)]
        occupancy_grids = [np.full((image_size, image_size, 3), 128, dtype=np.uint8) for _ in range(3)]  # Start with unknown (grey)

        for idx, scan in enumerate(self.scan_data):
            angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
            robot_x, robot_y = image_size // 2, image_size // 2  # Robot position in image coordinates

            for i, (angle, distance) in enumerate(zip(angles, scan.ranges)):
                # Adjust distance for out-of-range measurements to treat as free space
                # if distance == 0 or distance > scan.range_max:
                #     distance = scan.range_max
                old=self.old_scans[idx]
                old_distance= old.ranges[i]
                # Transform laser scan points to image coordinates
                x = int(robot_x + (distance * np.cos(angle)) * image_size / (scan.range_max * 2))
                y = int(robot_y + (distance * np.sin(angle)) * image_size / (scan.range_max * 2))


                

                # Draw a line for free space
                cv2.line(occupancy_grids[idx], (robot_x, robot_y), (x, y), (255, 255, 255), 1)

                # Check if the point is within the image bounds to mark as an obstacle
                if 0 <= x < image_size and 0 <= y < image_size:
                    if distance < scan.range_max:  # Only mark as an obstacle if not out-of-range
                        if i in glass_indices:
                            color = (0, 0, 255)
                            
                        else:
                            color = (0, 0, 0)
                            # Transform laser scan points to image coordinates
                            x = int(robot_x + (old_distance * np.cos(angle)) * image_size / (scan.range_max * 2))
                            y = int(robot_y + (old_distance * np.sin(angle)) * image_size / (scan.range_max * 2))
                        rgb_images[idx][y, x] = [0, 0, 0]
                        cv2.circle(occupancy_grids[idx], (x, y), 2, color, -1)


        # Convert ROS Time to datetime object and then to a string for the timestamp
        timestamp_str = datetime.datetime.fromtimestamp(scan_timestamp.to_sec()).strftime("%Y%m%d-%H%M%S")
        # Combine the images to create a 6-channel image
        multi_channel_image = np.concatenate((*rgb_images, *occupancy_grids), axis=2)

        # Save the multi-channel image as a NumPy array
        np.save(f"/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/multi_channel_images/{timestamp_str}.npy", multi_channel_image)

        # # Save or display images
        cv2.imwrite(f"/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/rgb_images/{timestamp_str}.jpg", rgb_images[0])
        cv2.imwrite(f"/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/occupancy_grids/{timestamp_str}.jpg", occupancy_grids[1])

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
                'orientation': yaw  # Radians
            }

            with open(f"/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/gt_poses/{timestamp_str}.json", 'w') as f:
                json.dump(pose_data, f)

if __name__ == '__main__':
    rospy.init_node('laser_scan_processor', anonymous=True)
    lsp = LaserScanProcessor()
    rospy.spin()
