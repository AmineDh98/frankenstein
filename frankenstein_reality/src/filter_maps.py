import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


base_directory = '/home/aminedhemaied/Thesis/results/map15'



def find_corners_by_color(image_path, color_threshold):
    # Load the image in color
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Make a copy for drawing
    image_with_corners = image.copy()

    # Split the image into its color channels
    blue_channel, green_channel, red_channel = cv2.split(image)

    # Find where the red channel is significantly higher than the green and blue channels
    red_threshold_mask = (red_channel > color_threshold[0]) & \
                         (green_channel < color_threshold[1]) & \
                         (blue_channel < color_threshold[2])

    # Convert the mask to uint8 format
    red_threshold_mask = np.uint8(red_threshold_mask) * 255

    # Find contours in the mask
    contours, _ = cv2.findContours(red_threshold_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract the center of the contour area which represents the corner
    corners = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            corners.append((cx, cy))
            # Draw a circle at the corner
            cv2.circle(image_with_corners, (cx, cy), radius=10, color=(0, 255, 0), thickness=2)

    # Convert image to RGB for matplotlib compatibility
    image_with_corners = cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB)

    # Display the image with drawn corners
    plt.imshow(image_with_corners)
    plt.show()

    print("Detected Corners:", corners)
    return corners

def apply_mask(image, corners):
    # Assume corners are the rectangle vertices
    mask = np.zeros_like(image)
    if corners:  # Check if corners list is not empty
        corners = np.array(corners, dtype=np.int32)
        cv2.fillPoly(mask, [corners], (255,))
    else:
        print("No corners detected, check corner detection settings.")
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def calculate_accuracy(ground_truth, slam_map):
    # Calculate accuracy
    correct_predictions = np.sum((ground_truth == slam_map) & (ground_truth > 0))
    total_predictions = np.sum(ground_truth > 0)  # Only consider non-unknown areas
    accuracy = correct_predictions / total_predictions if total_predictions else 0
    return accuracy

def calculate_iou(ground_truth, slam_map):
    # Calculate Intersection (True Positives)
    intersection = np.logical_and(ground_truth, slam_map)
    # Calculate Union (True Positives + False Positives + False Negatives)
    union = np.logical_or(ground_truth, slam_map)
    # Calculate IoU
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) else 0
    return iou_score


def calculate_confusion_matrix(ground_truth, slam_map, threshold=128):
    # Binarize images
    gt_binarized = (ground_truth > threshold).astype(int)
    slam_binarized = (slam_map > threshold).astype(int)

    # Flatten the matrices
    gt_flat = gt_binarized.flatten()
    slam_flat = slam_binarized.flatten()

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(gt_flat, slam_flat, labels=[0, 1])

    return conf_matrix


# Load the images with corners
ground_truth_path = f"{base_directory}/ground_truth_map_with_corners.png"
slam_map_path = f"{base_directory}/transformed_slam_map_with_corners.png"
ground_truth_with_corners = cv2.imread(ground_truth_path)
slam_map_with_corners = cv2.imread(slam_map_path)

corner_color_lower = np.array([200, 0, 100])  # Lower bound of pink in RGB
corner_color_upper = np.array([255, 200, 200])  # Upper bound of pink in RGB

color_threshold =  (205, 200, 200)  # Red channel needs to be greater than 150, green/blue less than 50


corners_gt = find_corners_by_color(ground_truth_path, color_threshold)
corners_slam = find_corners_by_color(slam_map_path, color_threshold)

if not corners_gt or not corners_slam:
    print("Error: No corners detected in one or both images. Exiting.")
else:
    masked_gt = apply_mask(cv2.cvtColor(ground_truth_with_corners, cv2.COLOR_BGR2GRAY), corners_gt)
    masked_slam = apply_mask(cv2.cvtColor(slam_map_with_corners, cv2.COLOR_BGR2GRAY), corners_slam)



    # Save the filtered images
    cv2.imwrite(f"{base_directory}/filtered_ground_truth.png", masked_gt)
    cv2.imwrite(f"{base_directory}/filtered_slam_map.png", masked_slam)

    # Display the filtered images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(masked_gt, cmap='gray')
    plt.title('Filtered Ground Truth')
    plt.subplot(1, 2, 2)
    plt.imshow(masked_slam, cmap='gray')
    plt.title('Filtered SLAM Map')
    plt.show()

    # Calculate accuracy
    accuracy = calculate_accuracy(masked_gt, masked_slam)
    print(f"Accuracy of SLAM map: {accuracy:.2%}")

    # Plot accuracy
    plt.figure(figsize=(6, 4))
    plt.bar(['SLAM Accuracy'], [accuracy], color='blue')
    plt.title('SLAM Map Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.show()

        # Calculate IoU
    iou_score = calculate_iou(masked_gt, masked_slam)
    print(f"IoU Score: {iou_score:.2%}")

    # Plot IoU
    plt.figure(figsize=(6, 4))
    plt.bar(['SLAM IoU'], [iou_score], color='green')
    plt.title('SLAM Map IoU Score')
    plt.ylabel('IoU Score')
    plt.ylim(0, 1)
    plt.show()


    # Calculate the confusion matrix
    conf_matrix = calculate_confusion_matrix(masked_gt, masked_slam)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Free (Predicted)', 'Occupied (Predicted)'],
                yticklabels=['Free (Actual)', 'Occupied (Actual)'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()


   