# Introduction
This repository is to be deployed on the Frankenstein AGV which is a prototype AGV robot developed by Tavil.
# Objective
The main objective of this project is to replace the PLC module with a ROS module that can perform the mapping and localization task. ROS will offer more flexibility and customization to the implemented system, allowing for the integration of a wide range of sensors and hardware components.
The PLC module output will be used as a ground truth to be compared to the implemented SLAM results for testing and evaluation purposes. 

# Task
The repository solves the mapping and localization challenge for an industrial AGV in a production environment. 
It includes:
* Frankenstein robot description (Gazebo model, geometry, etc...).
* Implementation of the cartographer SLAM including (Odometry computation, LIDAR data filtering).
* Implementation of a Glass detection algorithm.
* Solving the kidnapped robot problem using a deep learning approach.

## Created by
**Amine Dhemaied** -Intelligent field robotics master student

# Project architecture

<p align="center">
    <img src="images/overall.jpg" height="500" alt="Your image description">
</p>






AGV robot model in Gazebo with Cartographer SLAM implemented

To do mapping on Gazebo simulation run:
```
roslaunch frankenstein_gazebo frankenstein_gazebo.launch
```

To do pure localization on Gazebo simulation run:
```
roslaunch frankenstein_gazebo frankenstein_localization_gazebo.launch load_state_filename:=path/to/your/file.pbstream
```


To do mapping on real robot run:
```
roslaunch frankenstein_reality frankenstein_reality.launch
```

To do pure localization on real robot run:
```
roslaunch frankenstein_reality frankenstein_localozation_reality.launch map_file:=/path/to/your/file.pbstream
```

To save the map before ending the mapping, run the following commands:
```
rosservice call /finish_trajectory 0
rosservice call /write_state "{filename: '/path/to/your/file.pbstream'}"
rosrun cartographer_ros cartographer_pbstream_to_ros_map -map_filestem=/path/to/your/file.pbstream -pbstream_filename=/path/to/your/file.pbstream -resolution=0.05
```