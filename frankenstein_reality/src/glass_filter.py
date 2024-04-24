#!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg


class LiDARGlassFilter:
    def __init__(self):
        rospy.init_node('lidar_glass_filter')
        self.pub_glass = rospy.Publisher('/glass_cloud', PointCloud2, queue_size=10)
        self.pub_filtered = rospy.Publisher('/f_cloud', PointCloud2, queue_size=10)
        self.sub = rospy.Subscriber('/cloud', PointCloud2, self.cloud_callback)
        self.window_size = 11
        self.threshold = 2
        self.image_size = 448
        self.max_range = 100.0 
        

        cv2.destroyAllWindows()
    def detect_glass(self, ranges, intensities):
        std_devs = np.array([np.std(ranges[max(0, i-self.window_size//2):i+self.window_size//2+1]) for i in range(len(ranges))])
        glass_indices = []
        non_glass_indices = []
        starts = []
        ends = []

        for i in range(1, len(ranges) - 1):
            if std_devs[i] > self.threshold and ranges[i]<35:
                if ranges[i] > ranges[i-1] and intensities[i] < intensities[i-1]:
                    starts.append(i)
                elif ranges[i] < ranges[i-1] and intensities[i] > intensities[i-1]:
                    ends.append(i)

        print('start = ', len(starts))
        print('ends = ', len(ends))

        paired_starts_ends = []
        pair = []
        for i in range(len(ranges)):
            if pair == [] and i in starts:
                pair.append(i)
            elif len(pair)==1 and i in ends:
                pair.append(i)
            
            if len(pair)==2:
                if abs((pair[1]-pair[0]))<60:
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
        self.occupancy_grid = np.full((self.image_size, self.image_size, 3), 128, dtype=np.uint8)
        robot_x, robot_y = self.image_size // 2, self.image_size // 2

        for idx, (angle, distance) in enumerate(zip(angles, ranges)):
            if distance > self.max_range:
                distance = self.max_range

            x = int(robot_x + (distance * np.cos(angle)) * self.image_size / (2 * self.max_range))
            y = int(robot_y + (distance * np.sin(angle)) * self.image_size / (2 * self.max_range))

            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                cv2.line(self.occupancy_grid, (robot_x, robot_y), (y, x), (255, 255, 255), 2)
                if idx in glass_indices:
                    cv2.circle(self.occupancy_grid, (y, x), 3, (0, 0, 255), -1)
                else:
                    cv2.circle(self.occupancy_grid, (y, x), 3, (0, 255, 0), -1)

        cv2.imwrite('/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/occupancy_grid.png', self.occupancy_grid)


if __name__ == '__main__':
    try:
        filter_node = LiDARGlassFilter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
