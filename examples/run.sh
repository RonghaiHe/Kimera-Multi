#!/bin/bash

###
 # Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
 # @Author: Ronghai He
 # @Date: 2024-09-02 15:59:25
 # @LastEditors: RonghaiHe hrhkjys@qq.com
 # @LastEditTime: 2025-03-26 15:14:06
 # @FilePath: /src/kimera_multi/examples/run.sh
 # @Version: 
 # @Description: This script runs different SLAM examples based on the input argument.
 # @Example: ./run.sh 2 12_07 | ./run.sh | ./run.sh 0
### 

# Add hash table for NAME_TIME_ to strings
declare -A TIME2DATASET
TIME2DATASET=(
    ["10_14"]="campus_outdoor_10_14"
    ["12_07"]="campus_tunnels_12_07"
    ["12_08"]="campus_hybrid_12_08"
)    

CATKIN_WS_="/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws"
NAME_TIME_=${2:-"12_07"}
DATA_PATH_="/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/"${TIME2DATASET[$NAME_TIME_]}
LOG_DIR_=${DATA_PATH_}"/log_data_"$NAME_TIME_

# If no input, Run the example
# ./run.sh or ./run.sh 6 or ./run.sh 6 12_07
if [ $# -eq 0 -o "$1" = "6" ]; then
    # bash run.sh
    echo "No input, run the CSLAM example of " $NAME_TIME_
    CATKIN_WS=${CATKIN_WS_} \
    DATA_PATH=${DATA_PATH_} \
    LOG_DIR=${LOG_DIR_} \
    NAME_TIME=${NAME_TIME_} \
    tmuxp load 1014-example.yaml

# ./run.sh 0 or ./run.sh 0 12_08
elif [ $1 = "2" ]; then
    # bash run.sh 1
    echo "Run the CSLAM example with mapping for 2 robots of " $NAME_TIME_ 
    CATKIN_WS=${CATKIN_WS_} DATA_PATH=${DATA_PATH_} LOG_DIR=${LOG_DIR_} NAME_TIME=${NAME_TIME_} tmuxp load 1014-example2.yaml

# ./run.sh 1 or ./run.sh 1 12_08
elif [ $1 = "1" ]; then
    # bash run.sh 0
    echo "Run the single SLAM example for multi-robot of " $NAME_TIME_
    CATKIN_WS=${CATKIN_WS_} DATA_PATH=${DATA_PATH_} LOG_DIR=${LOG_DIR_} NAME_TIME=${NAME_TIME_} tmuxp load 1014-example_single6.yaml

# ./run.sh 0 or ./run.sh 0 12_08
elif [ $1 = "0" ]; then
    # bash run.sh 1
    echo "Run the single example for test mapping (single robot) of " $NAME_TIME_ 
    CATKIN_WS=${CATKIN_WS_} DATA_PATH=${DATA_PATH_} LOG_DIR=${LOG_DIR_} NAME_TIME=${NAME_TIME_} ROBOT_NAME="acl_jackal2" tmuxp load 1014-example_single_global_map.yaml

# ./run.sh 00 or ./run.sh 00 12_08
elif [ $1 = "00" ]; then
    # bash run.sh 1
    echo "Run the single example for test (single robot) of " $NAME_TIME_ 
    CATKIN_WS=${CATKIN_WS_} DATA_PATH=${DATA_PATH_} LOG_DIR=${LOG_DIR_} NAME_TIME=${NAME_TIME_} ROBOT_NAME="acl_jackal2" tmuxp load 1014-example_single.yaml
else
    echo "Invalid input"
fi

