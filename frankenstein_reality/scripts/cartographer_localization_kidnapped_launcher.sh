#!/bin/bash
MAP_FILE=$1
source /home/aminedhemaied/cartographer_ws/install_isolated/setup.bash
roslaunch cartographer_ros frankenstein_reality_localization_kidnapped.launch load_state_filename:=${MAP_FILE}

