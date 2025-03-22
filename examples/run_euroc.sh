#!/bin/bash

###
 # Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
 # @Author: Ronghai He
 # @Date: 2025-02-09 15:59:25
 # @LastEditors: RonghaiHe hrhkjys@qq.com
 # @LastEditTime: 2025-02-09 15:59:25
 # @FilePath: /src/kimera_multi/examples/run.sh
 # @Version: 
 # @Description: This script runs different SLAM examples for EuRoC based on the input argument.
 # @Example: ./run.sh 2 12_07 | ./run.sh | ./run.sh 0
### 

declare -A SCENE2DATASET
SCENE2DATASET=(
    ["MH_01"]="MH_01_easy"
    ["MH_02"]="MH_02_easy"
    ["MH_03"]="MH_03_medium"
    ["MH_04"]="MH_04_difficult"
    ["MH_05"]="MH_05_difficult"
    ["V1_01"]="V1_01_easy"
    ["V1_02"]="V1_02_medium"
    ["V1_03"]="V1_03_difficult"
    ["V2_01"]="V2_01_easy"
    ["V2_02"]="V2_02_medium"
    ["V2_03"]="V2_03_difficult"
)  

CATKIN_WS_="/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws"
NAME_SCENE_=${2:-"MH_01"}
DATA_PATH_="/media/sysu/Data/EuRoC/"${SCENE2DATASET[$NAME_SCENE_]}
LOG_DIR_=${DATA_PATH_}"/log_data_"$NAME_SCENE_

DIR_GT_="vicon0"
if [[ ${NAME_SCENE_:0:2} == "MH" ]]; then
    DIR_GT_="state_groundtruth_estimate0"
fi

# If no input, Run the example
# ./run.sh or ./run.sh 2 or ./run.sh 2 12_07
if [ $# -eq 0 -o "$1" = "2" ]; then
    # bash run.sh
    echo "No input, run the CSLAM example of " $NAME_SCENE_
    CATKIN_WS=${CATKIN_WS_} \
    DATA_PATH=${DATA_PATH_} \
    LOG_DIR=${LOG_DIR_} \
    DATASET=${SCENE2DATASET[$NAME_SCENE_]} \
    tmuxp load euroc_multi.yaml

# ./run.sh 1 or ./run.sh 1 12_08
elif [ $1 = "1" ]; then
    # bash run.sh 0
    echo "Run the single SLAM example for multi-robot of " $NAME_SCENE_
    CATKIN_WS=${CATKIN_WS_} \
    DATA_PATH=${DATA_PATH_} \
    LOG_DIR=${LOG_DIR_} \
    DATASET=${SCENE2DATASET[$NAME_SCENE_]} \
    tmuxp load euroc_single6.yaml

# ./run.sh 0 or ./run.sh 0 12_08
elif [ $1 = "0" ]; then
    # bash run.sh 1
    echo "Run the single example for test (single robot) of " $NAME_SCENE_ 
    CATKIN_WS=${CATKIN_WS_} \
    DATA_PATH=${DATA_PATH_} \
    LOG_DIR=${LOG_DIR_} \
    DATASET=${SCENE2DATASET[$NAME_SCENE_]} \
    DIR_GT=${DIR_GT_} \
    tmuxp load euroc_single.yaml
else
    echo "Invalid input"
fi

