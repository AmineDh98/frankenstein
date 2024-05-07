#!/bin/bash
MAP_FILE=$1
source /home/emin/cartographer_ws/install_isolated/setup.bash
roslaunch cartographer_ros frankenstein_reality_localization.launch load_state_filename:=${MAP_FILE}

