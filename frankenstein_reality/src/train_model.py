import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau



class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # Assuming 'n_channels' as the number of channels in the input image, you need to set this value.
        n_channels = 18 # You should replace this with the actual number of channels
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Assuming the size of the feature map before flattening is 128x1x1, which depends on input image size and pooling layers.
        self.fc1 = nn.Linear(in_features=128*1*1, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=3)  # The output layer for 3-DOF pose

    def forward(self, x):
        # Apply the convolutional layers with Leaky ReLU activations and pooling
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01))
        x = self.pool4(F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.01))
        x = self.pool5(F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.01))
        
        # Flatten the tensor
        x = torch.flatten(x, 1)
        
        # Dynamically compute the input size for the first fully connected layer
        # We'll only need to do this once, so we can check if fc1's weight has been set appropriately yet
        if self.fc1.weight.shape[1] != x.shape[1]:
            # Adjust the in_features of fc1
            self.fc1 = nn.Linear(x.shape[1], 4096).to(x.device)
        
        # Apply the fully connected layers
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.01)
        x = self.fc4(x)
        
        return x




class LidarPoseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'multi_channel_images')
        self.poses_dir = os.path.join(root_dir, 'gt_poses')

        # Assuming files in 'multi_channel_images' and 'gt_poses' match and are sorted
        self.file_list = [f for f in os.listdir(self.images_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.images_dir, self.file_list[idx])
        pose_name = os.path.join(self.poses_dir, self.file_list[idx].replace('.npy', '.json'))

        image = np.load(img_name)
        with open(pose_name, 'r') as f:
            pose_data = json.load(f)
            position = pose_data['position']
            orientation = pose_data['orientation']
            pose = np.array([position['x'], position['y'], orientation])

        sample = {'image': image, 'pose': pose}

        if self.transform:
            sample = self.transform(sample)

        return sample

# Define a transform to convert arrays to PyTorch tensors
class ToTensor(object):
    def __call__(self, sample):
        image, pose = sample['image'], sample['pose']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'pose': torch.from_numpy(pose).float()}


# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop
    
# Initialize the Dataset and DataLoader
train_dataset = LidarPoseDataset(
    root_dir='/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data/train',
    transform=ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = LidarPoseDataset(
    root_dir='/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data/val',
    transform=ToTensor()
)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

test_dataset = LidarPoseDataset(
    root_dir='/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/data/test',
    transform=ToTensor()
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Usually, we don't shuffle test data




# Custom loss function with separate position and orientation error components
def custom_loss(output, target, beta=1.0):
    # Position loss is the Euclidean distance between predicted and target positions
    position_loss = torch.norm(output[:, :2] - target[:, :2], dim=1).mean()
    
    # Orientation loss is the mean squared error of the angle differences
    # Assuming that the orientation angle is in radians and wrapped within [-π, π]
    orientation_error = output[:, 2] - target[:, 2]
    orientation_error = torch.where(
        orientation_error > np.pi, orientation_error - 2 * np.pi, orientation_error
    )
    orientation_error = torch.where(
        orientation_error < -np.pi, orientation_error + 2 * np.pi, orientation_error
    )
    orientation_loss = (orientation_error ** 2).mean()
    
    # Combine losses with the scaling factor beta for the orientation loss
    loss = position_loss + beta * orientation_loss
    return loss


# Create the CNN model
model = CustomCNN()
beta_value = 1.0 

# If you're using a GPU, move the model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Training loop with early stopping
def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=25, beta=beta_value):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs = data['image'].to(device)
            labels = data['pose'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = custom_loss(outputs, labels, beta)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs = data['image'].to(device)
                labels = data['pose'].to(device)
                outputs = model(inputs)
                loss = custom_loss(outputs, labels, beta)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

 # Manually print the learning rate if it was reduced
        if scheduler.get_last_lr()[0] < optimizer.param_groups[0]['lr']:
            print(f'Learning rate reduced to: {scheduler.get_last_lr()[0]}')

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')


        if early_stopping(val_loss):
            print("Early stopping")
            break

    print('Training complete')

# Run the training loop
train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=100, beta=beta_value)

# Run the training loop
train_model(model, train_loader, val_loader, optimizer, num_epochs=100, beta=beta_value)

# Save the trained model
torch.save(model.state_dict(), '/home/aminedhemaied/catkin_ws/src/frankenstein/frankenstein_reality/models/cnn_pose_estimator.pth')

# Evaluate the model on test data
model.eval()  # Set model to evaluation mode
test_loss = 0.0
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data['image'].to(device)
        labels = data['pose'].to(device)
        outputs = model(inputs)
        loss = custom_loss(outputs, labels)
        test_loss += loss.item()
print(f'Test Loss: {test_loss / len(test_loader)}')

