import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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



def load_model(model_path):
    model = CustomCNN()
    # Add map_location='cpu' to load the model to the CPU
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def preprocess_image(image_path):
    image = np.load(image_path)
    # Assuming model expects the image in CHW format
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dimension
    return image

def predict_position(model, image):
    with torch.no_grad():
        output = model(image)
    return output.squeeze().numpy()  # Convert to numpy array and remove batch dimension

def load_ground_truth(gt_path):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_position = np.array([gt_data['position']['x'], gt_data['position']['y']])
    gt_orientation = gt_data['orientation']
    return gt_position, gt_orientation

def plot_on_map(image_file, resolution, origin, predicted, ground_truth, ground_truth_orientation, arrow_length=30):
    img = plt.imread(image_file)
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray', origin='lower')
    orientation = predicted[2]
    # Convert positions to pixel coordinates
    predicted_pixel = ((predicted[:2] - origin[:2]) / resolution).astype(int)
    gt_pixel = ((ground_truth - origin[:2]) / resolution).astype(int)

    # Plotting positions
    ax.plot(predicted_pixel[0], predicted_pixel[1], 'ro', label='Predicted')
    ax.plot(gt_pixel[0], gt_pixel[1], 'go', label='Ground Truth')

    # Calculate the end point of the orientation arrow
    end_point_x = predicted_pixel[0] + arrow_length * np.cos(orientation)
    end_point_y = predicted_pixel[1] + arrow_length * np.sin(orientation)

    # Draw an arrow to show orientation
    ax.arrow(predicted_pixel[0], predicted_pixel[1], end_point_x-predicted_pixel[0], end_point_y-predicted_pixel[1], 
             head_width=3, head_length=5, fc='red', ec='red')
    
    # Calculate the end point of the orientation arrow
    end_point_x = gt_pixel[0] + arrow_length * np.cos(ground_truth_orientation)
    end_point_y = gt_pixel[1] + arrow_length * np.sin(ground_truth_orientation)

    # Draw an arrow to show orientation
    ax.arrow(gt_pixel[0], gt_pixel[1], end_point_x-gt_pixel[0], end_point_y-gt_pixel[1], 
             head_width=3, head_length=5, fc='green', ec='green')

    ax.legend()
    plt.show()

# Load model
model = load_model("/home/aminedhemaied/Downloads/models/8th/cnn_pose_estimatorNewOcc_noPar.pth")

# Preprocess the image
image = preprocess_image("/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data3/multi_channel_images/20240408-104930.npy")

# Predict the position using the model
predicted_position = predict_position(model, image)
print('predicted_position ', predicted_position)
# Load the ground truth
ground_truth_position, ground_truth_orientation = load_ground_truth("/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data3/gt_poses/20240408-104930.json")

# Load map parameters from the YAML file
with open("/home/aminedhemaied/bagfiles/light_bags/best1/map.yaml", 'r') as f:
    map_params = yaml.safe_load(f)
image_file = os.path.join(os.path.dirname(f.name), map_params['image'])
resolution = map_params['resolution']
origin = map_params['origin']

# Plot the predicted and ground truth positions on the map
plot_on_map(image_file, resolution, np.array(origin), predicted_position, ground_truth_position, ground_truth_orientation)