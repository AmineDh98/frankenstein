import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Load the images
gt_image_path = '/home/aminedhemaied/Thesis/photos/TAVIL_MAP_Sick.png'
slam_image_path = '/home/aminedhemaied/bagfiles/light_bags/best1/map.png'
gt_image = cv2.imread(gt_image_path, cv2.IMREAD_GRAYSCALE)
slam_image = cv2.imread(slam_image_path, cv2.IMREAD_GRAYSCALE)

def world_to_pixel(world_coordinates, resolution, origin):
    pixel_x = int((world_coordinates[0] - origin[0]) / resolution)
    pixel_y = int((world_coordinates[1] - origin[1]) / resolution)
    return (pixel_x, pixel_y)

# Function to draw corners on the image
def draw_corners(image, corners, color=(100, 55, 250), radius=50, thickness=-1):
    for corner in corners:
        cv2.circle(image, (int(corner[0]), int(corner[1])), radius, color, thickness)
    return image

origin_world = [216.41, 360.201, 0.0]  # SLAM origin in world coordinates
resolution_initial = 0.08  # initial resolution
origin_pixel_initial = world_to_pixel(origin_world[:2], resolution_initial, (0, 0))

print('origin_pixel_initial = ', origin_pixel_initial)


# Define corner points for each image (in pixels)
gt_corners = np.array([[372, 2493], [385, 575], [2376, 584], [2378, 2486]], dtype=np.float32)
slam_corners = np.array([ [2114, 2040], [4547, 2140], [4416, 4613], [2016, 4516]], dtype=np.float32)



# Draw corners on the original images
gt_image_with_corners = cv2.cvtColor(gt_image, cv2.COLOR_GRAY2BGR)  # Convert to BGR to draw colored points
slam_image_with_corners = cv2.cvtColor(slam_image, cv2.COLOR_GRAY2BGR)
gt_image_with_corners = draw_corners(gt_image_with_corners, gt_corners)
slam_image_with_corners = draw_corners(slam_image_with_corners, slam_corners)

slam_image_with_origin = draw_corners(slam_image_with_corners, [np.array(origin_pixel_initial)], color=(0, 255, 0), radius=30)


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

# Draw rescaled corners on the resized SLAM image
slam_resized_with_corners = cv2.cvtColor(slam_resized, cv2.COLOR_GRAY2BGR)
slam_resized_with_corners = draw_corners(slam_resized_with_corners, slam_corners_rescaled)

# Compute the transformation matrix to align the corners
transform_matrix = cv2.getPerspectiveTransform(slam_corners_rescaled, gt_corners)

# Warp the resized SLAM image using the transformation matrix
slam_transformed = cv2.warpPerspective(slam_resized, transform_matrix, (gt_image.shape[1], gt_image.shape[0]))

# Draw corresponding corners on the transformed SLAM image
slam_transformed_with_corners = cv2.cvtColor(slam_transformed, cv2.COLOR_GRAY2BGR)
slam_transformed_with_corners = draw_corners(slam_transformed_with_corners, gt_corners)

# Generate the error map
error_map = cv2.absdiff(gt_image, slam_transformed)


origin_pixel_transformed = cv2.perspectiveTransform(np.array([[origin_pixel_initial]], dtype=np.float32), transform_matrix)[0][0]
slam_transformed_with_origin = draw_corners(slam_transformed_with_corners, [origin_pixel_transformed], color=(0, 255, 0), radius=30)
resolution_transformed = resolution_initial / average_scale
print('resolution_transformed = ', resolution_transformed)
plt.subplot(221), plt.imshow(gt_image_with_corners), plt.title('Ground Truth Map with Corners')
plt.subplot(222), plt.imshow(slam_image_with_origin), plt.title('Initial SLAM Map with Origin')
plt.subplot(223), plt.imshow(slam_transformed_with_origin), plt.title('Transformed SLAM Map with Origin')
ax = plt.subplot(224), plt.imshow(error_map, cmap='Reds'), plt.title('Error Map')
plt.colorbar(ax[1], ax=ax[0], orientation='vertical')
plt.show()



# # Plot the images and the error map with a color bar
# plt.figure(figsize=(12, 10))
# plt.subplot(221), plt.imshow(gt_image_with_corners), plt.title('Ground Truth Map with Corners')
# plt.subplot(222), plt.imshow(slam_resized_with_corners), plt.title('Resized SLAM Map with Corners')
# plt.subplot(223), plt.imshow(slam_transformed_with_corners), plt.title('Transformed SLAM Map with Corners')
# ax = plt.subplot(224), plt.imshow(error_map, cmap='Reds'), plt.title('Error Map')
# plt.colorbar(ax[1], ax=ax[0], orientation='vertical')
# plt.show()

# base_directory = '/home/aminedhemaied/Thesis/results/map'

# # Save each image
# cv2.imwrite(f"{base_directory}/ground_truth_map_with_corners.png", gt_image_with_corners)
# cv2.imwrite(f"{base_directory}/resized_slam_map_with_corners.png", slam_resized_with_corners)
# cv2.imwrite(f"{base_directory}/transformed_slam_map_with_corners.png", slam_transformed_with_corners)
# cv2.imwrite(f"{base_directory}/error_map.png", error_map)
