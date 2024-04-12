import cv2
import numpy as np
import matplotlib.pyplot as plt


def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Load the images
gt_image_path = '/home/aminedhemaied/Thesis/photos/TAVIL_MAP_Sick.png'
slam_image_path = '/home/aminedhemaied/bagfiles/light_bags/best1/map_Resized.png'
gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
slam_image = cv2.imread(slam_image_path, cv2.IMREAD_GRAYSCALE)




# Define corner points for each image (in pixels)
gt_corners = np.array([[372, 2493], [385, 575], [2376, 584], [2378, 2486]], dtype=np.float32)
slam_corners = np.array([[890, 859], [1910, 900], [1861, 1943], [847, 1902]], dtype=np.float32)

# Calculate the average scale factor based on the distances between corner points
gt_distances = [calculate_distance(gt_corners[i], gt_corners[(i + 1) % 4]) for i in range(4)]
slam_distances = [calculate_distance(slam_corners[i], slam_corners[(i + 1) % 4]) for i in range(4)]
scale_factors = [gt_d / slam_d for gt_d, slam_d in zip(gt_distances, slam_distances)]
average_scale = sum(scale_factors) / len(scale_factors)

# Rescale SLAM image
new_dimensions = (int(slam_image.shape[1] * average_scale), int(slam_image.shape[0] * average_scale))
slam_resized = cv2.resize(slam_image, new_dimensions)



# Recalculate the SLAM corners based on the average scale
slam_corners_rescaled = np.array([corner * average_scale for corner in slam_corners], dtype=np.float32)

# Compute the transformation matrix to align the corners
transform_matrix = cv2.getPerspectiveTransform(slam_corners_rescaled, gt_corners)

# Warp the resized SLAM image using the transformation matrix
slam_transformed = cv2.warpPerspective(slam_resized, transform_matrix, (gt_image.shape[1], gt_image.shape[0]))

# Generate the error map
error_map = cv2.absdiff(gt_image, slam_transformed)

# Plot the images and the error map with a color bar
plt.figure(figsize=(12, 10))
plt.subplot(221), plt.imshow(gt_image, cmap='gray'), plt.title('Ground Truth Map')
plt.subplot(222), plt.imshow(slam_resized, cmap='gray'), plt.title('Resized SLAM Map')
plt.subplot(223), plt.imshow(slam_transformed, cmap='gray'), plt.title('Transformed SLAM Map')
ax = plt.subplot(224), plt.imshow(error_map, cmap='Reds'), plt.title('Error Map')
plt.colorbar(ax[1], ax=ax[0], orientation='vertical')
plt.show()

base_directory = '/home/aminedhemaied/Thesis/results/map'

# Save each image
cv2.imwrite(f"{base_directory}/ground_truth_map.png", gt_image)
cv2.imwrite(f"{base_directory}/resized_slam_map.png", slam_resized)
cv2.imwrite(f"{base_directory}/transformed_slam_map.png", slam_transformed)
cv2.imwrite(f"{base_directory}/error_map.png", error_map)


