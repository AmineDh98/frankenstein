import yaml

def generate_yaml_file(filename, image_filename, resolution, origin, negate, occupied_thresh, free_thresh):
    # Ensure that the data is in plain float format, not a NumPy or other complex type
    data = {
        'image': image_filename,
        'resolution': float(resolution),
        'origin': [float(origin[0]), float(origin[1]), 0.0],  # Ensure origin is a list of floats
        'negate': negate,
        'occupied_thresh': occupied_thresh,
        'free_thresh': free_thresh
    }
    
    # Write the YAML file with plain types
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

# Parameters - assuming you've calculated these correctly in previous parts of your code
image_filename_transformed = '/home/aminedhemaied/Thesis/results/map/transformed_slam_map.pgm'  # Change if needed
negate = 0
occupied_thresh = 0.65
free_thresh = 0.196
resolution_transformed = 0.06  # Example resolution, replace with your calculated value
new_origin_yaml = [-10.5, 15.3]  # Example origin coordinates in meters, replace with your calculated values

# Generate the YAML file
generate_yaml_file('/home/aminedhemaied/Thesis/results/map/transformed_slam_map.yaml', 
                   image_filename_transformed, resolution_transformed, new_origin_yaml, negate, occupied_thresh, free_thresh)

print("YAML file generated successfully with plain data types!")