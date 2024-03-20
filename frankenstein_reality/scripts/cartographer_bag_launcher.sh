#!/bin/bash
BAG_FILE=$1
source /home/aminedhemaied/cartographer_ws/install_isolated/setup.bash
roslaunch cartographer_ros demo_frankenstein_reality.launch bag_filename:=$BAG_FILE
