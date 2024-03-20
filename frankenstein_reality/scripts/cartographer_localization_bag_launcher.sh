#!/bin/bash
MAP_FILE=$1
BAG_FILE=$2

source /home/aminedhemaied/cartographer_ws/install_isolated/setup.bash
roslaunch cartographer_ros demo_frankenstein_reality_localization.launch load_state_filename:=${MAP_FILE} bag_filename:=${BAG_FILE}

