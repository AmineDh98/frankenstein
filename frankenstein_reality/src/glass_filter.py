#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sensor_msgs.msg import PointCloud2, LaserScan
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from sensor_msgs.msg import PointField


class LiDARGlassFilter:
    def __init__(self):
        rospy.init_node('lidar_glass_filter')
        self.angle_min = 0.0
        self.angle_max = 0.0
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.image_size = 448
        self.max_range = 50.0 

        self.occupancy_grid = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)



        # Set up the matplotlib figure and axes, based on image_size
        self.fig, self.ax = plt.subplots(figsize=(12, 12), dpi=100)
        
        self.im = self.ax.imshow(self.occupancy_grid)

        # Start the animation loop
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50)
        
        self.pub_glass = rospy.Publisher('/glass_cloud', PointCloud2, queue_size=10)
        self.pub_filtered = rospy.Publisher('/f_cloud', PointCloud2, queue_size=10)
        self.sub = rospy.Subscriber('/cloud', PointCloud2, self.cloud_callback)
        

    def update_plot(self, frame):
        # Update the image displayed on the plot
        self.im.set_data(self.occupancy_grid)
        return self.im,

    def scan_callback(self, msg):
        self.scan = msg.ranges
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max



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
        return glass_indices, non_glass_indices, ranges, intensities



 
    
    def cloud_callback(self, cloud_msg):
        ranges, intensities, angles = self.extract_ranges_intensities_angles(cloud_msg)
        glass_indices, non_glass_indices, updated_ranges, intensities = self.detect_glass(ranges, intensities)
        glass_cloud_msg = self.create_cloud(cloud_msg, glass_indices, updated_ranges, intensities)
        filtered_cloud_msg = self.create_cloud(cloud_msg, non_glass_indices, updated_ranges, intensities)
        self.pub_glass.publish(glass_cloud_msg)
        self.pub_filtered.publish(filtered_cloud_msg)
        self.generate_images_and_live_plot(angles, updated_ranges, glass_indices)

    def create_cloud(self, cloud_msg, indices, ranges, intensities):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = cloud_msg.header.frame_id

        # Read all points from the original cloud message
        points = list(pc2.read_points(cloud_msg, skip_nans=True))

        # Select and update points based on provided indices
        updated_points = []
        for i in indices:
            if i < len(points) and i < len(ranges) and i < len(intensities):
                # Extract original point
                p = points[i]
                # Update point with new range and intensity
                updated_point = (p[0], p[1], p[2], intensities[i], ranges[i])
                updated_points.append(updated_point)

        # Define fields for the new cloud, assuming the original fields include 'x', 'y', 'z', 'intensity', and 'range'
        fields = list(cloud_msg.fields)  # Make a copy of the existing fields
        fields.append(PointField(
            name='range',
            offset=fields[-1].offset + 4,  # Assuming all fields are 32-bit floats, hence the offset increase by 4 bytes
            datatype=PointField.FLOAT32,
            count=1
        ))

        # Create a new PointCloud2 message with the updated points
        filtered_cloud = pc2.create_cloud(header, fields, updated_points)
        return filtered_cloud





    def extract_ranges_intensities_angles(self, cloud_msg):
        ranges = []
        intensities = []
        angles = []
        for p in pc2.read_points(cloud_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            x, y, z, intensity = p
            range_val = np.sqrt(x**2 + y**2 + z**2)
            angle = np.arctan2(y, x)
            ranges.append(range_val)
            intensities.append(intensity)
            angles.append(angle)
        return np.array(ranges), np.array(intensities), np.array(angles)

    def generate_images_and_live_plot(self, angles, ranges, glass_indices):
        self.occupancy_grid.fill(128)  # Reset the grid to a gray background
        robot_x, robot_y = self.image_size // 2, self.image_size // 2

        for idx, (angle, distance) in enumerate(zip(angles, ranges)):
            # if distance > self.max_range:
            #     distance = self.max_range

            x = int(robot_x + (distance * np.cos(angle)) * self.image_size / (2 * self.max_range))
            y = int(robot_y + (distance * np.sin(angle)) * self.image_size / (2 * self.max_range))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                
                if idx in glass_indices:
                    color = (255, 255, 0)  # Yellow for in-between
                else:
                    color = (0, 255, 0)  # Green for non-glass indices
                self.occupancy_grid[x, y] = color  # Plot point as a colored pixel





if __name__ == '__main__':
    try:
        filter_node = LiDARGlassFilter()
        plt.show()  # This will block until the window is closed
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        plt.close()
