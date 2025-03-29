#!/bin/sh
###
 # Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
 # @Author: Ronghai He
 # @Date: 2025-03-29 15:44:26
 # @LastEditors: RonghaiHe hrhkjys@qq.com
 # @LastEditTime: 2025-03-29 15:45:43
 # @FilePath: /src/kimera_multi/examples/run_s3e.sh
 # @Version: 
 # @Description: 
 # 
### 
CATKIN_WS="/home/robot/kimera_multi_ws" 
LOG_DIR="/home/robot/dataset/S3E/log_data" 
DATA_PATH="/home/robot/dataset/S3E"
ROSBAG="v1_campus_road"

tmuxp load $CATKIN_WS/src/kimera_multi/examples/s3e.yaml
