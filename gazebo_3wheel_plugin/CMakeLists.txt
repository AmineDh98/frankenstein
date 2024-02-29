cmake_minimum_required(VERSION 3.0.2)
project(gazebo_3wheel_plugin)

## Compile as C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  nav_msgs
  tf
  tf2_ros
  gazebo_ros
  gazebo_msgs
)

# If you're using dynamic reconfigure, find it and include its configuration before catkin_package
# find_package(dynamic_reconfigure)
# generate_dynamic_reconfigure_options(
#   cfg/YourConfig.cfg
# )

find_package(gazebo REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)



link_directories(${GAZEBO_LIBRARY_DIRS})
include_directories(include ${Boost_INCLUDE_DIR} ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})



add_library(${PROJECT_NAME} src/gazebo_3wheel_plugin.cpp)


target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
  ${Boost_LIBRARIES}
)




catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME} 
  CATKIN_DEPENDS roscpp rospy std_msgs geometry_msgs nav_msgs tf gazebo_ros gazebo_msgs tf2_ros
  DEPENDS 
gazebo
  # Removed 'DEPENDS GAZEBO' because it's not correctly setting GAZEBO_INCLUDE_DIRS or GAZEBO_LIBRARIES
)