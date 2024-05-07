# import os

# # Paths to the folders
# folder1_path = '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data1scan/multi_channel_images'
# folder2_path = '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data1scan/gt_poses'

# # List files in each folder
# folder1_files = os.listdir(folder1_path)
# folder2_files = os.listdir(folder2_path)

# # Extract base names without extensions
# folder1_basenames = set(os.path.splitext(file)[0] for file in folder1_files)
# folder2_basenames = set(os.path.splitext(file)[0] for file in folder2_files)

# # Find files that are in folder1 but not in folder2, and vice versa
# extra_in_folder1 = folder1_basenames - folder2_basenames
# extra_in_folder2 = folder2_basenames - folder1_basenames

# # Remove the extra files
# for file in folder1_files:
#     if os.path.splitext(file)[0] in extra_in_folder1:
#         os.remove(os.path.join(folder1_path, file))
#         print(f'Removed {file} from {folder1_path}')

# for file in folder2_files:
#     if os.path.splitext(file)[0] in extra_in_folder2:
#         os.remove(os.path.join(folder2_path, file))
#         print(f'Removed {file} from {folder2_path}')

# print('Cleanup complete.')


# import os
# import shutil
# from random import sample

# def organize_data(src_folder1, src_folder2, base_path, n_val=800, n_test=500):
#     # Ensure base directories exist
#     for dtype in ['train', 'val', 'test']:
#         for subfolder in ['gt_poses', 'multi_channel_images']:
#             os.makedirs(os.path.join(base_path, dtype, subfolder), exist_ok=True)

#     # List files (assuming the names without extension match)
#     files = [os.path.splitext(f)[0] for f in os.listdir(src_folder1)]
    
#     # Randomly select files for validation and test sets
#     val_files = sample(files, n_val)
#     remaining_files = list(set(files) - set(val_files))
#     test_files = sample(remaining_files, n_test)
#     train_files = list(set(remaining_files) - set(test_files))

#     # Function to move files
#     def move_files(file_list, src1, src2, dest1, dest2):
#         for base_name in file_list:
#             shutil.move(os.path.join(src1, base_name + '.npy'), os.path.join(dest1, base_name + '.npy'))
#             shutil.move(os.path.join(src2, base_name + '.json'), os.path.join(dest2, base_name + '.json'))
    
#     # Move files to respective directories
#     move_files(val_files, src_folder1, src_folder2, os.path.join(base_path, 'val', 'multi_channel_images'), os.path.join(base_path, 'val', 'gt_poses'))
#     move_files(test_files, src_folder1, src_folder2, os.path.join(base_path, 'test', 'multi_channel_images'), os.path.join(base_path, 'test', 'gt_poses'))
#     move_files(train_files, src_folder1, src_folder2, os.path.join(base_path, 'train', 'multi_channel_images'), os.path.join(base_path, 'train', 'gt_poses'))

#     print('Data organization complete.')

# # Define the source folders
# src_folder1 = '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/multi_channel_images'
# src_folder2 = '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/gt_poses'

# # Define the base path for the train/val/test structure
# base_path = '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass'

# # Organize the data
# organize_data(src_folder1, src_folder2, base_path)





import os
import json

def is_in_parking_area(x, y):
    return 0 <= x <= 10 and -10 <= y <= 10

def update_json_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Check the position and set parking value
            x = data['position']['x']
            y = data['position']['y']
            data['parking'] = is_in_parking_area(x, y)
            
            # Write back the modified data to the file
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)

# Paths to the directories
directories = [
    '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/train/gt_poses',
    '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/val/gt_poses',
    '/home/emin/catkin_ws/src/frankenstein/frankenstein_reality/data_glass/test/gt_poses'
]

for dir_path in directories:
    update_json_files(dir_path)


