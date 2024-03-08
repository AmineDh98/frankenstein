### Frankenstein
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