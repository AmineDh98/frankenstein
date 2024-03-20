#!/bin/bash
source /home/aminedhemaied/sick_scan_ws/devel_isolated/setup.bash
roslaunch sick_scan_xd sick_nav_31x.launch hostname:=172.16.1.3 frame_id:=lidar_link range_max:=200.0
