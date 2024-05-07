# import os
# import json

# # Define the source and destination directories
# source_dir = '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data3/gt_poses'
# destination_dir = '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data3/gt_poses_parking'

# # Make the destination directory if it doesn't already exist
# if not os.path.exists(destination_dir):
#     os.makedirs(destination_dir)

# # Function to check if a position is in the parking area
# def in_parking_area(x, y):
#     return 0 <= x <= 9 and -9 <= y <= 9

# # Process each file in the source directory
# for filename in os.listdir(source_dir):
#     if filename.endswith('.json'):
#         # Construct the full file paths
#         src_path = os.path.join(source_dir, filename)
#         dest_path = os.path.join(destination_dir, filename)
        
#         # Read the JSON data
#         with open(src_path, 'r') as file:
#             data = json.load(file)
        
#         # Check if the position is within the parking area
#         x = data['position']['x']
#         y = data['position']['y']
#         data['parking'] = in_parking_area(x, y)
        
#         # Write the modified data to the new file
#         with open(dest_path, 'w') as file:
#             json.dump(data, file, indent=4)

import torch
print(torch.cuda.is_available())
