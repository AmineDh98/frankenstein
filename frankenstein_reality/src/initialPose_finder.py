#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
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
        self.fc4 = nn.Linear(512, 3)  # Output layer for 3-DOF pose
        
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
        dummy_input = torch.zeros(1, 18, 448, 448)
        output = self._forward_features(dummy_input)
        self._num_flat_features = int(output.nelement() / output.shape[0])  # Ensure this is an int
        # Dynamically set the in_features for fc1 based on the calculated number of flat features
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
        self.scan_subscriber = rospy.Subscriber("/scan", LaserScan, self.scan_callback)
    def scan_callback(self, data):
        self.scans_received += 1
        self.scan_data.append(data)
        if self.scans_received == 3:
            self.process_data()
            self.scans_received = 0
            self.scan_data = []

    def process_data(self):
        if self.counter ==0:
            image = self.generate_image()
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
        if (len(self.predicted_positions) == 3):
            avg_position = np.mean(self.predicted_positions, axis=0)
            print('Averaged position:', avg_position)
            self.publish_initial_pose(avg_position[0])
            self.sender.send_initial_pose(avg_position[0])
            self.plot_on_map(avg_position[0])
        

    def plot_on_map(self, predicted, arrow_length=100):
        # Load map parameters from the YAML file
        with open("/home/aminedhemaied/bagfiles/light_bags/best1/map.yaml", 'r') as f:
            map_params = yaml.safe_load(f)
        image_file = os.path.join(os.path.dirname(f.name), map_params['image'])
        resolution = map_params['resolution']
        origin = map_params['origin']

        img = plt.imread(image_file)
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray', origin='lower')
        orientation = predicted[2]
        # Convert positions to pixel coordinates
        predicted_pixel = ((predicted[:2] - origin[:2]) / resolution).astype(int)
        

        # Plotting positions
        ax.plot(predicted_pixel[0], predicted_pixel[1], 'ro', label='Predicted')
        # Calculate the end point of the orientation arrow
        end_point_x = predicted_pixel[0] + arrow_length * np.cos(orientation)
        end_point_y = predicted_pixel[1] + arrow_length * np.sin(orientation)
        # Draw an arrow to show orientation
        ax.arrow(predicted_pixel[0], predicted_pixel[1], end_point_x-predicted_pixel[0], end_point_y-predicted_pixel[1], 
                head_width=3, head_length=5, fc='red', ec='red')
        ax.legend()
        plt.show()
        self.counter = 1
    def generate_image(self):
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

        # Combine the images to create a 6-channel image
        self.multi_channel_image = np.concatenate((*rgb_images, *occupancy_grids), axis=2)
        return self.multi_channel_image






def load_model():
    model = CustomCNN()
    model.load_state_dict(torch.load('/home/aminedhemaied/Downloads/models/8th/cnn_pose_estimatorNewOcc_noPar.pth', map_location='cpu'))
    model.eval()
    return model



if __name__ == '__main__':
    rospy.init_node('laser_scan_processor', anonymous=True)
    model = load_model()
    sender = PoseSender()
    lsp = LaserScanProcessor(model, sender)
    
    rospy.spin()
