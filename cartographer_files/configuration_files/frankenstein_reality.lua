-- Copyright 2016 The Cartographer Authors
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

include "map_builder.lua"
include "trajectory_builder.lua"

options = {
  map_builder = MAP_BUILDER,
  trajectory_builder = TRAJECTORY_BUILDER,
  map_frame = "world",
  tracking_frame = "robot_frame",
  published_frame = "robot_frame",
  odom_frame = "odom",
  provide_odom_frame = true,
  publish_frame_projected_to_2d = true,
  use_pose_extrapolator = true,
  use_odometry = true,
  use_nav_sat = false,
  use_landmarks = false,
  num_laser_scans = 0,
  num_multi_echo_laser_scans = 0,
  num_subdivisions_per_laser_scan = 10,
  num_point_clouds = 1,
  lookup_transform_timeout_sec = 0.2,
  submap_publish_period_sec = 0.3,
  pose_publish_period_sec = 5e-3,
  trajectory_publish_period_sec = 30e-3,
  rangefinder_sampling_ratio = 1.,
  odometry_sampling_ratio = 1.,
  fixed_frame_pose_sampling_ratio = 1.,
  imu_sampling_ratio = 1.,
  landmarks_sampling_ratio = 1.,
  publish_to_tf = true,
  publish_tracked_pose = true,
}

MAP_BUILDER.use_trajectory_builder_2d = true

TRAJECTORY_BUILDER_2D.num_accumulated_range_data = 20 --1
TRAJECTORY_BUILDER_2D.use_imu_data = false
TRAJECTORY_BUILDER_2D.max_range = 200
TRAJECTORY_BUILDER_2D.min_range = 0.5
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 200.0 --5.0
TRAJECTORY_BUILDER_2D.min_z =1.0
TRAJECTORY_BUILDER_2D.max_z =5.0
MAP_BUILDER.num_background_threads = 8--4
TRAJECTORY_BUILDER_2D.voxel_filter_size = 0.025 --0.025
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.occupied_space_weight=1.0 --1.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.translation_weight = 2e2 --10.0
TRAJECTORY_BUILDER_2D.ceres_scan_matcher.rotation_weight = 4e2 --40.0

TRAJECTORY_BUILDER_2D.submaps.grid_options_2d.resolution = 0.05 --0.05
TRAJECTORY_BUILDER_2D.submaps.num_range_data = 40 --90

TRAJECTORY_BUILDER_2D.use_online_correlative_scan_matching = true



TRAJECTORY_BUILDER_2D.adaptive_voxel_filter = {
    max_length = 0.5, --0.5
    min_num_points = 200, --200
    max_range = 50., --50.0
}

TRAJECTORY_BUILDER_2D.loop_closure_adaptive_voxel_filter = {
  max_length = 0.9, --0.9
  min_num_points = 100, --100
  max_range = 50., --50.0
}


POSE_GRAPH.optimize_every_n_nodes = 90 --90


POSE_GRAPH.optimization_problem.odometry_translation_weight = 1e3 --1e5
POSE_GRAPH.optimization_problem.odometry_rotation_weight = 1 --1e5
POSE_GRAPH.optimization_problem.local_slam_pose_translation_weight=1e5 --1e5
POSE_GRAPH.optimization_problem.local_slam_pose_rotation_weight=1e5 --1e5

POSE_GRAPH.constraint_builder.loop_closure_translation_weight = 1.1e4 --1.1e4
POSE_GRAPH.constraint_builder.loop_closure_rotation_weight = 1e5 --1e5

POSE_GRAPH.matcher_translation_weight = 5e2--5e2
POSE_GRAPH.matcher_rotation_weight = 1.6e3--1.6e3



POSE_GRAPH.global_sampling_ratio =0.0025 --0.003
POSE_GRAPH.constraint_builder.max_constraint_distance = 15.0 --15.0
POSE_GRAPH.constraint_builder.sampling_ratio= 0.25 --0.3
POSE_GRAPH.constraint_builder.min_score= 0.45 --0.55


POSE_GRAPH.global_constraint_search_after_n_seconds = 10.0 --10.

POSE_GRAPH.optimization_problem.huber_scale = 1e1 --1e1





POSE_GRAPH.max_num_final_iterations = 300 --200





return options
