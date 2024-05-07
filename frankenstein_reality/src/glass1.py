#!/usr/bin/env python3
import rospy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sensor_msgs.msg import PointCloud2, LaserScan
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg


class LiDARGlassFilter:
    def __init__(self):
        rospy.init_node('lidar_glass_filter')
        self.pub_glass = rospy.Publisher('/glass_cloud', PointCloud2, queue_size=10)
        self.pub_filtered = rospy.Publisher('/f_cloud', PointCloud2, queue_size=10)
        self.sub = rospy.Subscriber('/cloud', PointCloud2, self.cloud_callback)

        self.image_size = 4480
        self.max_range = 250.0

        self.window_size = 5
        self.threshold = 0.5 #6
        self.image_size = 448
        self.max_range = 70.0 
        self.intensity_min = 0
        self.intensity_max = 10000
        self.intens = 4500
        self.max_width = 30
        self.min_width = 10

        # Set up the matplotlib figure and axes, based on image_size
        self.fig, self.ax = plt.subplots(figsize=(10, 10), dpi=100)
        self.occupancy_grid = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        self.im = self.ax.imshow(self.occupancy_grid)

        # Start the animation loop
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=50)

    def update_plot(self, frame):
        # Update the image displayed on the plot
        self.im.set_data(self.occupancy_grid)
        return self.im,

    def scan_callback(self, msg):
        self.scan = msg.ranges



        

    def detect_glass(self, ranges, intensities):
        std_devs = np.array([np.std(ranges[max(0, i-self.window_size//2):i+self.window_size//2+1]) for i in range(len(ranges))])
        glass_indices = []
        non_glass_indices = []
        starts = []
        ends = []
        edge = []
        counter=0
        for i in range(1, len(ranges) - 1):
            # if std_devs[i] > self.threshold or ranges[i]>200:
            if  std_devs[i] > self.threshold : # and self.intens>intensities[i]  or ranges[i]>150
                counter+=1
                if ranges[i] > ranges[i-1] and intensities[i] < intensities[i-1] : #ranges[i] > ranges[i-1] and 
                    starts.append(i)
                elif ranges[i] < ranges[i-1] and intensities[i] > intensities[i-1] : # ranges[i] < ranges[i-1] and
                    ends.append(i)
        # print('filter 1 number = ', counter)

        # print('start = ', len(starts))
        # print('ends = ', len(ends))

        paired_starts_ends = []
        pair = []
        for i in range(len(ranges)):
            if pair == [] and i in starts:
                pair.append(i)
            elif len(pair)==1 and i in ends:
                pair.append(i)
            
            if len(pair)==2:
                if self.min_width < abs((pair[1]-pair[0]))<self.max_width:
                    paired_starts_ends.append((pair[0], pair[1]))
                pair=[]

        for start, end in paired_starts_ends:
            for idx in range(start, end + 1):
                glass_indices.append(idx)
                
        non_glass_indices = list(set(range(len(ranges))) - set(glass_indices))

        for start, end in paired_starts_ends:
            if end > start:
                D1 = ranges[start]
                D2 = ranges[end]
                N = end - start + 1
                Uc = (D2 - D1) / N
                if N<30 and abs(D1-D2)<5:
                    for i in range(start + 1, end):
                        ranges[i] = D1 + Uc * (i - start)


        return glass_indices, non_glass_indices, ranges
    
    def create_cloud(self, cloud_msg, indices, ranges):
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = cloud_msg.header.frame_id
        points = list(pc2.read_points(cloud_msg, skip_nans=True))
        filtered_points = [points[i] for i in indices if i < len(points)]
        updated_points = [(p[0], p[1], p[2], ranges[i]) if i < len(ranges) else p for i, p in enumerate(filtered_points)]
        filtered_cloud = pc2.create_cloud(header, cloud_msg.fields, updated_points)
        return filtered_cloud

    def cloud_callback(self, cloud_msg):
        ranges, intensities, angles = self.extract_ranges_intensities_angles(cloud_msg)
        glass_indices, non_glass_indices, updated_ranges = self.detect_glass(ranges, intensities)
        glass_cloud_msg = self.create_cloud(cloud_msg, glass_indices, updated_ranges)
        filtered_cloud_msg = self.create_cloud(cloud_msg, non_glass_indices, updated_ranges)
        self.pub_glass.publish(glass_cloud_msg)
        self.pub_filtered.publish(filtered_cloud_msg)
        self.generate_images_and_live_plot(angles, updated_ranges, glass_indices)


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
            if distance > self.max_range:
                distance = self.max_range

            x = int(robot_x + (distance * np.cos(angle)) * self.image_size / (2 * self.max_range))
            y = int(robot_y + (distance * np.sin(angle)) * self.image_size / (2 * self.max_range))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                color = (0, 0, 255) if idx in glass_indices else (0, 255, 0)
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