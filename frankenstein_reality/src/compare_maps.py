import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt


def load_and_rescale_maps(gt_path, slam_path, gt_mm_dims, slam_mm_dims):
    # Load the ground truth and SLAM maps
    gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    slam_map = cv2.imread(slam_path, cv2.IMREAD_GRAYSCALE)
    
    # Calculate target size based on physical dimensions and desired resolution
    # Assuming the desired resolution is based on the ground truth map's resolution
    gt_resolution = gt_mm_dims[0] / gt_map.shape[1], gt_mm_dims[1] / gt_map.shape[0] # pixels/mm
    target_size = int(slam_mm_dims[0] * gt_resolution[0]), int(slam_mm_dims[1] * gt_resolution[1])
    
    # Resize SLAM map to match the physical size of the ground truth map
    slam_map_resized = cv2.resize(slam_map, target_size, interpolation=cv2.INTER_AREA)
    
    # Ensure both maps are the same size for comparison
    gt_map_resized = cv2.resize(gt_map, (slam_map_resized.shape[1], slam_map_resized.shape[0]), interpolation=cv2.INTER_AREA)
    
    return gt_map_resized, slam_map_resized

def compare_maps(gt_map, slam_map):
    # Calculate SSIM between the two maps
    score, _ = ssim(gt_map, slam_map, full=True)
    return score



# Path and physical dimensions of the maps
gt_path = '/home/aminedhemaied/Thesis/photos/TAVIL_MAP_Sick.png'
slam_path = '/home/aminedhemaied/bagfiles/light_bags/best1/map.pgm'
gt_mm_dims = (575.8, 605.4) # Width x Height in millimeters
slam_mm_dims = (2391.8, 2394.7) # Width x Height in millimeters

# Load the ground truth and SLAM maps
gt_map = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
slam_map = cv2.imread(slam_path, cv2.IMREAD_GRAYSCALE)

# Detect ORB features and compute descriptors.

orb = cv2.ORB_create()
keypoints_gt, descriptors_gt = orb.detectAndCompute(gt_map, None)
keypoints_slam, descriptors_slam = orb.detectAndCompute(slam_map, None)

# Function to apply the affine transformation to the SLAM map
def transform_map(slam_map, gt_corners, slam_corners):
    # Compute the affine transformation matrix
    transform_matrix = cv2.getAffineTransform(np.float32(slam_corners), np.float32(gt_corners))

    # Apply the affine transformation
    transformed_map = cv2.warpAffine(slam_map, transform_matrix, (gt_map.shape[1], gt_map.shape[0]), flags=cv2.INTER_LINEAR)
    
    return transformed_map, transform_matrix

def find_corners(image):
    # Convert to grayscale if it is not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply a Gaussian blur to smooth out the edges
    gray = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Find the contours in the edge map
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming the largest contour is the outline of the map
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    # Approximate the contour to a polygon
    epsilon = 0.1 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # If we have 4 points, then we have found our corners
    if len(approx) == 4:
        return approx.reshape((4, 2))
    else:
        # Here we would need to implement a fallback strategy
        return None

# Placeholder for the actual corner coordinates on the ground truth and SLAM maps
# You need to find these coordinates manually and replace these placeholder values
# The coordinates are in the format: [[x1, y1], [x2, y2], [x3, y3]] for the top-left, top-right, and bottom-left corners respectively
gt_corners = np.array([[372, 2493], [385, 575], [2376, 584]]) # Replace with actual ground truth corner coordinates
slam_corners = np.array([[2028, 4664], [2139, 2232], [4613, 2363]]) # Replace with actual SLAM corner coordinates



# Apply the transformation
# transformed_slam_map, transformation_matrix = transform_map(slam_map, gt_corners, slam_corners)

# # Display the original SLAM map, the transformed SLAM map, and the ground truth map
# plt.figure(figsize=(20, 10))

# plt.subplot(1, 3, 1)
# plt.imshow(slam_map, cmap='gray')
# plt.title('Original SLAM Map')

# plt.subplot(1, 3, 2)
# plt.imshow(transformed_slam_map, cmap='gray')
# plt.title('Transformed SLAM Map')

# plt.subplot(1, 3, 3)
# plt.imshow(gt_map, cmap='gray')
# plt.title('Ground Truth Map')

# plt.show()

# # Print the transformation matrix
# print("Transformation matrix:\n", transformation_matrix)

# Find corners in both images
gt_corners = find_corners(gt_map)
slam_corners = find_corners(slam_map)

# If corners are found in both images, proceed to find the homography
if gt_corners is not None and slam_corners is not None:
    # The coordinates need to be in float32 format for the function
    gt_corners = gt_corners.astype(np.float32)
    slam_corners = slam_corners.astype(np.float32)
    
    # Find the perspective transformation between the corners
    h, status = cv2.findHomography(slam_corners, gt_corners)
    
    # Apply this perspective transformation to the SLAM map
    height, width = gt_map.shape[:2]
    aligned_slam_map = cv2.warpPerspective(slam_map, h, (width, height))
    
    # Show the aligned SLAM map and the ground truth map
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(aligned_slam_map, cv2.COLOR_BGR2RGB))
    plt.title('Aligned SLAM Map')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(gt_map, cv2.COLOR_BGR2RGB))
    plt.title('Ground Truth Map')

    plt.show()
    
    # Print the transformation matrix
    print("Transformation matrix:\n", h)
else:
    print("Could not find four corners in both maps.")


# # Match descriptors using KNN.
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# matches = bf.match(descriptors_gt, descriptors_slam)
# # Sort them in the order of their distance (the lower the better).
# matches = sorted(matches, key=lambda x: x.distance)
# # Draw top matches
# img_matches = cv2.drawMatches(gt_map, keypoints_gt, slam_map, keypoints_slam, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(img_matches), plt.show()

# # Assuming enough matches are found, estimate an affine transformation
# if len(matches) > 10:
#     # Extract location of good matches
#     points_gt = np.zeros((len(matches), 2), dtype=np.float32)
#     points_slam = np.zeros((len(matches), 2), dtype=np.float32)
#     for i, match in enumerate(matches):
#         points_gt[i, :] = keypoints_gt[match.queryIdx].pt
#         points_slam[i, :] = keypoints_slam[match.trainIdx].pt
#     # Find affine transformation
#     matrix, mask = cv2.estimateAffinePartial2D(points_slam, points_gt, method=cv2.RANSAC, ransacReprojThreshold=5.0)
#     # Apply transformation to the SLAM map
#     transformed_slam_map = cv2.warpAffine(slam_map, matrix, (gt_map.shape[1], gt_map.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
#     # Display the original SLAM map, the transformed SLAM map, and the ground truth map
#     plt.figure(figsize=(20, 10))
#     plt.subplot(1, 3, 1)
#     plt.imshow(slam_map, cmap='gray')
#     plt.title('Original SLAM Map')
#     plt.subplot(1, 3, 2)
#     plt.imshow(transformed_slam_map, cmap='gray')
#     plt.title('Transformed SLAM Map')
#     plt.subplot(1, 3, 3)
#     plt.imshow(gt_map, cmap='gray')
#     plt.title('Ground Truth Map')
#     plt.show()
#     # Print the transformation matrix
#     print("Transformation matrix:\n", matrix)
# else:
#     print("Not enough matches were found - at least 10 are required, only {} were found".format(len(matches)))
#     transformed_slam_map = None
#     matrix = None
# # Save the transformed map for further comparison if required
# transformed_slam_map_path = '/mnt/data/transformed_slam_map.png'
# if transformed_slam_map is not None:
#     cv2.imwrite(transformed_slam_map_path, transformed_slam_map)
# # Provide the path for reference
# transformed_slam_map_path
# # Load and rescale maps
# gt_map, slam_map = load_and_rescale_maps(gt_path, slam_path, gt_mm_dims, slam_mm_dims)
# # Compare the maps
# similarity_score = compare_maps(gt_map, slam_map)
# print(f"Similarity score (SSIM) between the ground truth map and SLAM map: {similarity_score}")
