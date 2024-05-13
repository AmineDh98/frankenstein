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




include "frankenstein_reality.lua"

TRAJECTORY_BUILDER_2D.max_range = 50
TRAJECTORY_BUILDER_2D.missing_data_ray_length = 50.0 --5.0
TRAJECTORY_BUILDER.pure_localization_trimmer = {
  max_submaps_to_keep = 3,
}

POSE_GRAPH.optimize_every_n_nodes = 10
POSE_GRAPH.constraint_builder.global_localization_min_score = 0.45 --0.6
MAP_BUILDER.num_background_threads = 8--4
POSE_GRAPH.global_sampling_ratio =0.002 --0.003
POSE_GRAPH.constraint_builder.sampling_ratio= 0.2 --0.3
POSE_GRAPH.global_constraint_search_after_n_seconds = 10. --10.
POSE_GRAPH.constraint_builder.max_constraint_distance = 15.0 --15.0
POSE_GRAPH.constraint_builder.min_score= 0.55 --0.55
return options

