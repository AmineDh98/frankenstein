#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs.point_cloud2 as pc2
import numpy as np
import cv2
import datetime
import json
import tf

class PointCloudProcessor:
    def __init__(self):

        self.angle_min = 0.0
        self.angle_max = 0.0

        self.cloud_subscriber = rospy.Subscriber("/cloud", PointCloud2, self.cloud_callback)
        self.odom_subscriber = rospy.Subscriber("/tracked_pose", PoseStamped, self.odom_callback)
        
        self.clouds_received = 0
        self.old_clouds_received = 0
        self.cloud_data = []
        self.old_clouds = []

        self.last_odom = None
        self.last_odom_stamp = rospy.Time(0)

        self.window_size = 9
        self.threshold = 6
        self.image_size = 448
        self.max_range = 50.0
        self.intensity_min = 0
        self.intensity_max = 10000
        self.intens = 4500
        self.width = 35


    def odom_callback(self, data):
        self.last_odom = data
        self.last_odom_stamp = data.header.stamp

    def cloud_callback(self, msg):
        ranges, angles, intensities = self.process_cloud(msg)
        self.angle_min = np.min(angles) if len(angles) > 0 else 0
        self.angle_max = np.max(angles) if len(angles) > 0 else 0
        glass_indices, non_glass_indices, updated_ranges = self.detect_glass(ranges, intensities)
        old_cloud = msg
        self.coordinate(msg, old_cloud, glass_indices, ranges, angles, intensities)

    def process_cloud(self, cloud_msg):
        # Convert the PointCloud2 message to an array of coordinates
        gen = pc2.read_points(cloud_msg, skip_nans=True, field_names=("x", "y", "z", "intensity"))
        coords = np.array(list(gen))

        x, y, z, intensities = coords[:,0], coords[:,1], coords[:,2], coords[:,3]
        angles = np.arctan2(y, x)
        ranges = np.sqrt(x**2 + y**2)

        return ranges, angles, intensities

    def coordinate(self, cloud_msg, old_cloud, glass_indices, ranges, angles, intensities):
        if self.last_odom_stamp.to_sec() > 0 and abs((cloud_msg.header.stamp - self.last_odom_stamp).to_sec()) < 0.1:
            self.clouds_received += 1
            self.old_clouds_received += 1

            if self.clouds_received <= 3:
                self.cloud_data.append({'ranges': ranges, 'angles': angles, 'intensities': intensities})
                self.old_clouds.append(old_cloud)
                if self.clouds_received == 3:
                    self.generate_images_and_save_pose(cloud_msg.header.stamp, glass_indices)
                    self.clouds_received = 0
                    self.old_clouds_received = 0

                    self.cloud_data = []
                    self.old_clouds = []

 

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
    
    
   


    def generate_images_and_save_pose(self, cloud_timestamp, glass_indices):
        # Initialize the images with a resolution of 448x448
        image_size = 448
        rgb_images = [np.full((image_size, image_size, 3), 255, dtype=np.uint8) for _ in range(3)]
        occupancy_grids = [np.full((image_size, image_size, 3), 128, dtype=np.uint8) for _ in range(3)]  # Start with unknown (grey)

        # Compute robot position in image coordinates (assuming center of the image)
        robot_x, robot_y = image_size // 2, image_size // 2

        for idx, cloud in enumerate(self.cloud_data):
            ranges = cloud['ranges']
            angles = cloud['angles']
            intensities = cloud['intensities']

            for i, (angle, distance) in enumerate(zip(angles, ranges)):
                # Convert polar coordinates (angle, distance) to Cartesian coordinates for image
                x = int(robot_x + (distance * np.cos(angle)) * image_size / (self.max_range * 2))
                y = int(robot_y + (distance * np.sin(angle)) * image_size / (self.max_range * 2))

                # Draw a line for free space
                cv2.line(occupancy_grids[idx], (robot_x, robot_y), (x, y), (255, 255, 255), 1)

                # Check if the point is within the image bounds to mark as an obstacle
                if 0 <= x < image_size and 0 <= y < image_size:
                    if distance < self.max_range:  # Only mark as an obstacle if not out-of-range
                        color = (0, 0, 255) if i in glass_indices else (0, 0, 0)
                        cv2.circle(rgb_images[idx], (x, y), 2, color, -1)
                        cv2.circle(occupancy_grids[idx], (x, y), 2, color, -1)

        # Convert ROS Time to datetime object and then to a string for the timestamp
        timestamp_str = datetime.datetime.fromtimestamp(cloud_timestamp.to_sec()).strftime("%Y%m%d-%H%M%S")
        # Combine the images to create a 6-channel image
        multi_channel_image = np.concatenate((*rgb_images, *occupancy_grids), axis=2)

        # Save the multi-channel image as a NumPy array
        np.save(f"/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/multi_channel_images/{timestamp_str}.npy", multi_channel_image)

        # Save or display images
        cv2.imwrite(f"/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/rgb_images/{timestamp_str}.jpg", rgb_images[0])
        cv2.imwrite(f"/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/occupancy_grids/{timestamp_str}.jpg", occupancy_grids[1])

        # Save pose data with the same timestamp if it's recent enough
        if self.last_odom and abs((cloud_timestamp - self.last_odom.header.stamp).to_sec()) < 0.1:
            orientation_q = self.last_odom.pose.orientation
            _, _, yaw = tf.transformations.euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

            pose_data = {
                'timestamp': cloud_timestamp.to_sec(),  # Use the ROS Time object directly here
                'position': {
                    'x': self.last_odom.pose.position.x,
                    'y': self.last_odom.pose.position.y,
                },
                'orientation': yaw  # Radians
            }

            with open(f"/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/gt_poses/{timestamp_str}.json", 'w') as f:
                json.dump(pose_data, f)


if __name__ == '__main__':
    rospy.init_node('point_cloud_processor', anonymous=True)
    pcp = PointCloudProcessor()
    rospy.spin()
