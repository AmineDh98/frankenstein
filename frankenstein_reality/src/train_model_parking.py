import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau



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





class LidarPoseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'multi_channel_images')
        self.poses_dir = os.path.join(root_dir, 'gt_poses')
        self.file_list = [f for f in os.listdir(self.images_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.file_list[idx])
        pose_name = os.path.join(self.poses_dir, self.file_list[idx].replace('.npy', '.json'))

        image = np.load(img_name)
        with open(pose_name, 'r') as f:
            pose_data = json.load(f)
            position = pose_data['position']
            orientation = pose_data['orientation']
            parking = float(pose_data['parking'])  # Convert boolean to float (0.0 or 1.0)
            pose = np.array([position['x'], position['y'], orientation, parking])

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
    
def custom_loss(output, target, orientation_beta=1.0, pose_beta=1.0, parking_beta=1.0):
    position_loss = torch.norm(output[:, :2] - target[:, :2], dim=1).mean()
    orientation_error = output[:, 2] - target[:, 2]
    orientation_error = torch.where(
        orientation_error > np.pi, orientation_error - 2 * np.pi, orientation_error
    )
    orientation_error = torch.where(
        orientation_error < -np.pi, orientation_error + 2 * np.pi, orientation_error
    )
    orientation_loss = (orientation_error ** 2).mean()
    parking_output = output[:, 3]  
    parking_target = target[:, 3]
    
    parking_loss = F.binary_cross_entropy_with_logits(parking_output, parking_target)
    
    total_loss = position_loss * pose_beta + orientation_beta * orientation_loss + parking_beta * parking_loss
    return total_loss, position_loss, orientation_loss, parking_loss
    


    
def evaluate_model(model, test_loader, orientation_beta=1.0, pose_beta=1.0, parking_beta=1.0):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    test_position_loss = 0.0
    test_orientation_loss = 0.0
    test_parking_loss = 0.0
    total_samples = 0
    correct_parking_predictions = 0

    runningPar_lossPar = 0
    runningPar_position_lossPar = 0
    runningPar_orientation_lossPar = 0
    runningPar_parking_lossPar = 0

    correct_parking, parking_correct_count = 0, 0

    with torch.no_grad():
        for data in test_loader:
            inputs = data['image'].to(device)
            labels = data['pose'].to(device)
            outputs = model(inputs)
            
            loss, position_loss, orientation_loss, parking_loss = custom_loss(outputs, labels, orientation_beta, pose_beta, parking_beta)
            test_loss += loss.item()
            test_position_loss += position_loss.item()
            test_orientation_loss += orientation_loss.item()
            test_parking_loss += parking_loss.item()

            # Predict parking status
            parking_predictions = torch.sigmoid(outputs[:, 3]) > 0.5  # Convert logits to probabilities and classify
            correct_parking_predictions += (parking_predictions == labels[:, 3].bool()).sum().item()
            total_samples += labels.size(0)

            correct_indices = (parking_predictions == labels[:, 3].bool())
            if correct_indices.any():
                parking_correct_count += correct_indices.sum().item()
                lossPar, position_lossPar, orientation_lossPar, parking_lossPar = custom_loss(outputs[correct_indices], labels[correct_indices], orientation_beta, pose_beta, parking_beta)
                
                runningPar_lossPar += lossPar.item() * correct_indices.sum().item()
                runningPar_position_lossPar += position_lossPar.item() * correct_indices.sum().item()
                runningPar_orientation_lossPar += orientation_lossPar.item() * correct_indices.sum().item()
                runningPar_parking_lossPar += parking_lossPar.item() * correct_indices.sum().item()

    avg_test_loss = test_loss / len(test_loader)
    avg_test_position_loss = test_position_loss / len(test_loader)
    avg_test_orientation_loss = test_orientation_loss / len(test_loader)
    avg_test_parking_loss = test_parking_loss / len(test_loader)
    parking_accuracy = correct_parking_predictions / total_samples

    # Averages for correct parking predictions
    if parking_correct_count > 0:
        avg_runnungPar_lossPar = runningPar_lossPar / parking_correct_count
        avg_runnungPar_position_lossPar = runningPar_position_lossPar / parking_correct_count
        avg_runnungPar_orientation_lossPar = runningPar_orientation_lossPar / parking_correct_count
        avg_runnungPar_parking_lossPar = runningPar_parking_lossPar / parking_correct_count
    else:
        avg_runnungPar_lossPar, avg_runnungPar_position_lossPar, avg_runnungPar_orientation_lossPar, avg_runnungPar_parking_lossPar = 0, 0, 0, 0


    print(f'Test Loss: {avg_test_loss:.4f}, Position Loss: {avg_test_position_loss:.4f}, Orientation Loss: {avg_test_orientation_loss:.4f}, Parking Loss: {avg_test_parking_loss:.4f}')
    print(f'Parking Accuracy: {parking_accuracy:.4f}')
    print(f'Test Correct Parking - Avg Loss: {avg_runnungPar_lossPar:.4f}, Position Loss: {avg_runnungPar_position_lossPar:.4f}, Orientation Loss: {avg_runnungPar_orientation_lossPar:.4f}, Parking Loss: {avg_runnungPar_parking_lossPar:.4f}')


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=25, orientation_beta=1.0, pose_beta=1.0, parking_beta=1.0):
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss, running_position_loss, running_orientation_loss, running_parking_loss = 0.0, 0.0, 0.0, 0.0
        runningPar_lossPar, runningPar_position_lossPar, runningPar_orientation_lossPar, runningPar_parking_lossPar = 0.0, 0.0, 0.0, 0.0
        correct_parking, parking_correct_count = 0, 0
        
        train_correct, train_total = 0, 0

        # Training phase
        for data in train_loader:
            inputs = data['image'].to(device)
            labels = data['pose'].to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss, position_loss, orientation_loss, parking_loss = custom_loss(outputs, labels, orientation_beta, pose_beta, parking_beta)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_position_loss += position_loss.item()
            running_orientation_loss += orientation_loss.item()
            running_parking_loss += parking_loss.item()

            parking_pred = torch.sigmoid(outputs[:, 3]) > 0.5
            train_correct += (parking_pred == labels[:, 3].bool()).sum().item()
            train_total += labels.size(0)

            # Separate loss calculation for correct parking predictions
            correct_indices = (parking_pred == labels[:, 3].bool())
            if correct_indices.any():
                parking_correct_count += correct_indices.sum().item()
                lossPar, position_lossPar, orientation_lossPar, parking_lossPar = custom_loss(outputs[correct_indices], labels[correct_indices], orientation_beta, pose_beta, parking_beta)
                
                runningPar_lossPar += lossPar.item() * correct_indices.sum().item()
                runningPar_position_lossPar += position_lossPar.item() * correct_indices.sum().item()
                runningPar_orientation_lossPar += orientation_lossPar.item() * correct_indices.sum().item()
                runningPar_parking_lossPar += parking_lossPar.item() * correct_indices.sum().item()

        train_parking_accuracy = train_correct / train_total

        # Average losses
        avg_train_loss = running_loss / len(train_loader)
        avg_position_loss = running_position_loss / len(train_loader)
        avg_orientation_loss = running_orientation_loss / len(train_loader)
        avg_parking_loss = running_parking_loss / len(train_loader)

        # Averages for correct parking predictions
        if parking_correct_count > 0:
            avg_runnungPar_lossPar = runningPar_lossPar / parking_correct_count
            avg_runnungPar_position_lossPar = runningPar_position_lossPar / parking_correct_count
            avg_runnungPar_orientation_lossPar = runningPar_orientation_lossPar / parking_correct_count
            avg_runnungPar_parking_lossPar = runningPar_parking_lossPar / parking_correct_count
        else:
            avg_runnungPar_lossPar, avg_runnungPar_position_lossPar, avg_runnungPar_orientation_lossPar, avg_runnungPar_parking_lossPar = 0, 0, 0, 0


        print(f'Epoch {epoch+1}/{num_epochs}')
        
        print(f'Train Parking Accuracy: {train_parking_accuracy:.4f}')
        print(f'Train loss: {avg_train_loss:.4f}, Position Loss: {avg_position_loss:.4f}, Orientation Loss: {avg_orientation_loss:.4f}, Parking Loss: {avg_parking_loss:.4f}')
        print(f'Train Correct Parking - Avg Loss: {avg_runnungPar_lossPar:.4f}, Position Loss: {avg_runnungPar_position_lossPar:.4f}, Orientation Loss: {avg_runnungPar_orientation_lossPar:.4f}, Parking Loss: {avg_runnungPar_parking_lossPar:.4f}')
         
          # Validation phase
        model.eval()
        val_loss, val_position_loss, val_orientation_loss, val_parking_loss = 0.0, 0.0, 0.0, 0.0
        valPar_lossPar, valPar_position_lossPar, valPar_orientation_lossPar, valPar_parking_lossPar = 0.0, 0.0, 0.0, 0.0
        val_correct, val_total, val_parking_correct_count = 0, 0, 0

        with torch.no_grad():
            for data in val_loader:
                inputs = data['image'].to(device)
                labels = data['pose'].to(device)
                
                outputs = model(inputs)
                loss, position_loss, orientation_loss, parking_loss = custom_loss(outputs, labels, orientation_beta, pose_beta, parking_beta)
                
                val_loss += loss.item()
                val_position_loss += position_loss.item()
                val_orientation_loss += orientation_loss.item()
                val_parking_loss += parking_loss.item()

                parking_pred = torch.sigmoid(outputs[:, 3]) > 0.5
                val_correct += (parking_pred == labels[:, 3].bool()).sum().item()
                val_total += labels.size(0)

                # Separate loss calculation for correct parking predictions
                correct_indices = (parking_pred == labels[:, 3].bool())
                if correct_indices.any():
                    val_parking_correct_count += correct_indices.sum().item()
                    lossPar, position_lossPar, orientation_lossPar, parking_lossPar = custom_loss(outputs[correct_indices], labels[correct_indices], orientation_beta, pose_beta, parking_beta)
                    
                    valPar_lossPar += lossPar.item() * correct_indices.sum().item()
                    valPar_position_lossPar += position_lossPar.item() * correct_indices.sum().item()
                    valPar_orientation_lossPar += orientation_lossPar.item() * correct_indices.sum().item()
                    valPar_parking_lossPar += parking_lossPar.item() * correct_indices.sum().item()

        val_parking_accuracy = val_correct / val_total

        # Average validation losses
        avg_val_loss = val_loss / len(val_loader)
        avg_val_position_loss = val_position_loss / len(val_loader)
        avg_val_orientation_loss = val_orientation_loss / len(val_loader)
        avg_val_parking_loss = val_parking_loss / len(val_loader)

        # Averages for correct parking predictions in validation
        if val_parking_correct_count > 0:
            avg_valPar_lossPar = valPar_lossPar / val_parking_correct_count
            avg_valPar_position_lossPar = valPar_position_lossPar / val_parking_correct_count
            avg_valPar_orientation_lossPar = valPar_orientation_lossPar / val_parking_correct_count
            avg_valPar_parking_lossPar = valPar_parking_lossPar / val_parking_correct_count
        else:
            avg_valPar_lossPar, avg_valPar_position_lossPar, avg_valPar_orientation_lossPar, avg_valPar_parking_lossPar = 0, 0, 0, 0

        print(f'Validation Parking Accuracy: {val_parking_accuracy:.4f}')
        print(f'Validation loss: {avg_val_loss:.4f}, Position Loss: {avg_val_position_loss:.4f}, Orientation Loss: {avg_val_orientation_loss:.4f}, Parking Loss: {avg_val_parking_loss:.4f}')
        print(f'Validation Correct Parking - Avg Loss: {avg_valPar_lossPar:.4f}, Position Loss: {avg_valPar_position_lossPar:.4f}, Orientation Loss: {avg_valPar_orientation_lossPar:.4f}, Parking Loss: {avg_valPar_parking_lossPar:.4f}')
        
        # Scheduler step (consider what metric you want to base the scheduler step on, this is just an example)
        scheduler.step(avg_val_loss)

        # Early stopping check
        if early_stopping(avg_val_loss):
            print("Early stopping triggered")
            break



orientation_beta_value = 5 
pose_beta_value = 3
parking_beta_value = 1
number_of_epochs = 100
learning_rate = 1e-4
batchSize = 10
evaluationMode = True

# Initialize the Dataset and DataLoader
train_dataset = LidarPoseDataset(
    root_dir='/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/train',
    transform=ToTensor()
)
train_loader = DataLoader(train_dataset, batch_size=batchSize, shuffle=True)

val_dataset = LidarPoseDataset(
    root_dir='/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/val',
    transform=ToTensor()
)
val_loader = DataLoader(val_dataset, batch_size=batchSize, shuffle=True)

test_dataset = LidarPoseDataset(
    root_dir='/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/test',
    transform=ToTensor()
)
test_loader = DataLoader(test_dataset, batch_size=batchSize, shuffle=False)  # Usually, we don't shuffle test data

if evaluationMode==False:
    # Create the CNN model
    model = CustomCNN()
    

    # If you're using a GPU, move the model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop with early stopping and separate loss reporting

    # Run the training loop
    train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs=number_of_epochs, orientation_beta=orientation_beta_value, pose_beta=pose_beta_value, parking_beta=parking_beta_value)

    # Save the trained model
    torch.save(model.state_dict(), '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/models/glass.pth')

    

else:
    # Initialize the model
    

    # If you're using a GPU, move the model to GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CustomCNN().to(device)

    # Load the trained model parameters
    model.load_state_dict(torch.load('/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/models/glass.pth', map_location=device))

    # Ensure the model is in evaluation mode
    model.eval()

    evaluate_model(model, test_loader, orientation_beta=orientation_beta_value, pose_beta=pose_beta_value, parking_beta=parking_beta_value)




