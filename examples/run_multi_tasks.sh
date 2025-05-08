#!/bin/bash
###
 # Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
 # @Author: Ronghai He
 # @Date: 2025-04-14 18:47:59
 # @LastEditors: RonghaiHe hrhkjys@qq.com
 # @LastEditTime: 2025-04-18 02:02:02
 # @FilePath: /src/kimera_multi/examples/run_multi_tasks.sh
 # @Version: 
 # @Description: 
 # 
### 

bash run_multi_times.sh 2 12_08
bash run_multi_times.sh 2 12_07
bash run_multi_times.sh 1 12_07


# sleep 2200
cd /data/herh/kimera_multi_ws/src/kimera_multi/examples
# bash run_multi_times.sh 2 12_08
# bash run_multi_times.sh 1 12_08

bash run_multi_times.sh 1 12_08
bash run_multi_times.sh 1 12_07

# mv /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_outdoor_10_14/exp_single_/* /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_outdoor_10_14/exp_single_kf_undistort
# mkdir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_outdoor_10_14/exp_single_

# mv /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_tunnels_12_07/exp_distributed_ /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_tunnels_12_07/exp_distributed_mono_count_kf_undistort
# mkdir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_tunnels_12_07/exp_distributed_

# bash run_multi_times.sh 1 12_08
# mv /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_single_/* /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_single_kf_undistort
# mkdir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_single_

# bash run_multi_times.sh 2 12_08
# mv /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_distributed_ /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_distributed_mono_count_kf_undistort
# mkdir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_distributed_