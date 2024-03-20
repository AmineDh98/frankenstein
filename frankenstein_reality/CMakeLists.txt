cmake_minimum_required(VERSION 3.0.2)
project(frankenstein_reality)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_ros
  pcl_conversions
  laser_geometry

)

find_package(Boost REQUIRED COMPONENTS system)

## Declare a catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES robot_data_publisher
  CATKIN_DEPENDS roscpp sensor_msgs nav_msgs tf2 tf2_ros tf2_geometry_msgs pcl_ros pcl_conversions laser_geometry
  DEPENDS Boost
)

## Include headers
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(data_parser src/datapublisher.cpp)

## Add cmake target dependencies of the executable
add_dependencies(data_parser ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(data_parser
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)


add_executable(data_parser2 src/merge.cpp)

## Add cmake target dependencies of the executable
add_dependencies(data_parser2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(data_parser2
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(odometry_node_reality_dr src/odometry_reality_dead_reckoning.cpp)
target_link_libraries(odometry_node_reality_dr ${catkin_LIBRARIES})

add_executable(odometry_node_reality src/odometry_reality.cpp)
target_link_libraries(odometry_node_reality ${catkin_LIBRARIES})

add_executable(visualizer_node_reality src/visualizer_reality.cpp)
target_link_libraries(visualizer_node_reality ${catkin_LIBRARIES})

add_executable(scan_filter_node src/scan_filter.cpp)
target_link_libraries(scan_filter_node ${catkin_LIBRARIES})