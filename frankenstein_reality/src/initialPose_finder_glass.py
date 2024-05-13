#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import sensor_msgs.point_cloud2 as pc2
import cv2
import datetime
import tf.transformations
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from cartographer_ros_msgs.srv import StartTrajectory, StartTrajectoryRequest


class CustomCNN(nn.Module):
    def __init__(self, n_channels=18):
        super(CustomCNN, self).__init__()
        
        # Assuming 'n_channels' as the number of channels in the input image
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool6 = nn.MaxPool2d(2, 2)

        self.conv7 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(128)
        self.pool7 = nn.MaxPool2d(2, 2)

        # Placeholder for the number of input features to the first fully connected layer
        self._num_flat_features = None

        self.fc1 = nn.Linear(128 * 14 * 14, 4096)  # Placeholder value, will be dynamically set
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 4)  # Output layer for 3-DOF pose
        
        # Dummy input to set _num_flat_features
        self._set_num_flat_features()

    def forward(self, x):
        # Convolutional and pooling layers
        x = self._forward_features(x)
        
        # Fully connected layers
        x = x.view(-1, self._num_flat_features)  # Flatten the output for the fully connected layer
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.fc4(x)
        
        return x
    
    def _forward_features(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01))
        x = self.pool4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01))
        x = self.pool5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01))
        x = self.pool6(F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.01))
        x = self.pool7(F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.01))
        return x

    def _set_num_flat_features(self):
        # Recalculate the number of flat features
        dummy_input = torch.zeros(1, 18, 448, 448)
        output = self._forward_features(dummy_input)
        self._num_flat_features = int(output.nelement() / output.shape[0])
        # Dynamically set the in_features for fc1
        self.fc1 = nn.Linear(self._num_flat_features, 4096)

class PoseSender:
    def __init__(self):
        # Wait for the service to be available
        rospy.wait_for_service('/start_trajectory')
        try:
            self.start_trajectory = rospy.ServiceProxy('/start_trajectory', StartTrajectory)
        except rospy.ServiceException as exc:
            rospy.logerr("Service initialization failed: %s" % exc)

    def send_initial_pose(self, pose):
    # Prepare the pose in the service request format
        try:
            quaternion = tf.transformations.quaternion_from_euler(0, 0, pose[2])
            # Create an empty service request
            request = StartTrajectoryRequest()
            
            # Populate the request fields
            request.configuration_directory = '/home/aminedhemaied/cartographer_ws/install_isolated/share/cartographer_ros/configuration_files'
            request.configuration_basename = 'frankenstein_reality_localization.lua'
            request.use_initial_pose = True
            request.initial_pose.position.x = pose[0]
            request.initial_pose.position.y = pose[1]
            request.initial_pose.position.z = 0.0
            request.initial_pose.orientation.x = quaternion[0]
            request.initial_pose.orientation.y = quaternion[1]
            request.initial_pose.orientation.z = quaternion[2]
            request.initial_pose.orientation.w = quaternion[3]
            request.relative_to_trajectory_id = 0

            # Call the service
            response = self.start_trajectory(request)
            rospy.loginfo("Service call successful, response: %s" % response)
        except rospy.ServiceException as exc:
            rospy.logerr("Service call failed: %s" % exc)



class LaserScanProcessor:
    def __init__(self, model, sender):
        self.model = model
        self.sender = sender
        self.pose_publisher = rospy.Publisher("/cnn_pose", PoseStamped, queue_size=10)
        self.scans_received = 0
        self.scan_data = []
        self.predicted_positions=[]
        self.counter = 0
        self.angle_min = 0.0
        self.angle_max = 0.0

        self.cloud_subscriber = rospy.Subscriber("/cloud", PointCloud2, self.cloud_callback)
        self.clouds_received = 0
        self.old_clouds_received = 0
        self.cloud_data = []
        self.old_clouds = []

        self.window_size = 9
        self.threshold = 6
        self.image_size = 448
        self.max_range = 50.0
        self.intensity_min = 0
        self.intensity_max = 10000
        self.intens = 4500
        self.width = 35
        

        # Load map parameters from the YAML file
        with open("/home/aminedhemaied/bagfiles/map15/map15.yaml", 'r') as f:
            map_params = yaml.safe_load(f)
        self.image_file = os.path.join(os.path.dirname(f.name), map_params['image'])
        self.resolution = map_params['resolution']
        self.origin = map_params['origin']
        self.img = plt.imread(self.image_file)
        self.fig, self.ax = plt.subplots()

    
    def cloud_callback(self, msg):
        ranges, angles, intensities = self.process_cloud(msg)
        self.angle_min = np.min(angles) if len(angles) > 0 else 0
        self.angle_max = np.max(angles) if len(angles) > 0 else 0
        glass_indices, non_glass_indices, updated_ranges = self.detect_glass(ranges, intensities)
        old_cloud = msg
        print(self.cloud_data)
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
        self.clouds_received += 1
        self.old_clouds_received += 1

        if self.clouds_received <= 3:
            self.cloud_data.append({'ranges': ranges, 'angles': angles, 'intensities': intensities})
            self.old_clouds.append(old_cloud)
            if self.clouds_received == 3:
                self.process_data(cloud_msg.header.stamp, glass_indices)
                self.clouds_received = 0
                self.old_clouds_received = 0
                
                self.cloud_data = []
                self.old_clouds = []
    
    def process_data(self, stamp, glass_indices):
        if self.counter ==0:
            image = self.generate_images_and_save_pose(stamp, glass_indices)
            
            self.predict_and_plot(image)

    def publish_initial_pose(self, pose):
        initial_pose = PoseStamped()

        initial_pose.header.seq = 1
        initial_pose.header.stamp = rospy.Time.now()
        initial_pose.header.frame_id = "world"

        initial_pose.pose.position.x = pose[0]
        initial_pose.pose.position.y = pose[1]
        initial_pose.pose.position.z = 0.0

        quaternion = tf.transformations.quaternion_from_euler(0, 0, pose[2])
        initial_pose.pose.orientation.x = quaternion[0]
        initial_pose.pose.orientation.y = quaternion[1]
        initial_pose.pose.orientation.z = quaternion[2]
        initial_pose.pose.orientation.w = quaternion[3]

        self.pose_publisher.publish(initial_pose)

    def predict_and_plot(self, image):
        # Transpose the image to have channel as the second dimension (PyTorch format)
        image_tensor = torch.from_numpy(image.transpose((2, 0, 1))).float().unsqueeze(0)
        predicted_position = self.model(image_tensor).detach().numpy()
        print('Predicted position:', predicted_position)
        self.predicted_positions.append(predicted_position)
        if (len(self.predicted_positions) == 4):
            avg_position = np.mean(self.predicted_positions[:3], axis=0)
            print('Averaged position:', avg_position)
            self.publish_initial_pose(avg_position[0])
            self.sender.send_initial_pose(avg_position[0])
            # self.plot_on_map(avg_position[0])
        

    def plot_on_map(self, predicted, arrow_length=100):
        self.ax.imshow(self.img, cmap='gray', origin='lower')
        orientation = predicted[2]
        # Convert positions to pixel coordinates
        predicted_pixel = ((predicted[:2] - self.origin[:2]) / self.resolution).astype(int)
        

        # Plotting positions
        self.ax.plot(predicted_pixel[0], predicted_pixel[1], 'ro', label='Predicted')
        # Calculate the end point of the orientation arrow
        end_point_x = predicted_pixel[0] + arrow_length * np.cos(orientation)
        end_point_y = predicted_pixel[1] + arrow_length * np.sin(orientation)
        # Draw an arrow to show orientation
        self.ax.arrow(predicted_pixel[0], predicted_pixel[1], end_point_x-predicted_pixel[0], end_point_y-predicted_pixel[1], 
                head_width=3, head_length=5, fc='red', ec='red')
        self.ax.legend()
        plt.show()
        self.counter = 1
    



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

        # Combine the images to create a 6-channel image
        self.multi_channel_image = np.concatenate((*rgb_images, *occupancy_grids), axis=2)
        return self.multi_channel_image
    
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

def load_model(model_path):
    model = CustomCNN()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model



if __name__ == '__main__':
    rospy.init_node('laser_scan_processor', anonymous=True)

    # Get the model path from ROS parameter server
    model_path = rospy.get_param('~model_path', '/home/aminedhemaied/Downloads/models/glass1.pth')  # Default path as fallback
    model = load_model(model_path)

    sender = PoseSender()
    lsp = LaserScanProcessor(model, sender)
    
    rospy.spin()
