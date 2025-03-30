#!/bin/sh
DATA_PATH_="/home/robot/dataset/self"
LOG_DIR_=${DATA_PATH_}"/log_data_"$(date +"%Y_%m_%d_%H_%M_%S")
CATKIN_WS_="/home/robot/kimera_multi_ws" 

DATA_PATH=${DATA_PATH_} LOG_DIR=${LOG_DIR_} CATKIN_WS=${CATKIN_WS_} tmuxp load ${CATKIN_WS_}/src/kimera_multi/examples/test3.yaml
