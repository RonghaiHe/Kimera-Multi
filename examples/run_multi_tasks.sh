#!/bin/bash

bash run_multi_times.sh 1 12_07

mv /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_outdoor_10_14/test_single_/* /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_outdoor_10_14/test_single_kf_undistort
# mkdir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_outdoor_10_14/test_single_

# mv /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_tunnels_12_07/test_distributed_ /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_tunnels_12_07/test_distributed_mono_count_kf_undistort
# mkdir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_tunnels_12_07/test_distributed_

bash run_multi_times.sh 1 12_08
mv /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_hybrid_12_08/test_single_/* /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_hybrid_12_08/test_single_kf_undistort
mkdir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_hybrid_12_08/test_single_

# bash run_multi_times.sh 2 12_08
# mv /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_hybrid_12_08/test_distributed_ /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_hybrid_12_08/test_distributed_mono_count_kf_undistort
# mkdir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_hybrid_12_08/test_distributed_