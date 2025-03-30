#!/bin/sh
CATKIN_WS_="/home/robot/kimera_multi_ws" 
DATA_PATH_="/home/robot/dataset/self" 

if [ $# -eq 0 -o "$1" = "c" ]; then
    echo "Dataset collection"
    CATKIN_WS=$CATKIN_WS_ \
    DATA_PATH=$DATA_PATH_ \
    tmuxp load $CATKIN_WS_/src/kimera_multi/examples/data_collection.yaml
    
elif [ "$1" = "r" ]; then
    echo "Run CSLAM in real-world"    
    CATKIN_WS=$CATKIN_WS_ \
    LOG_DIR=${DATA_PATH_}/log_data \
    tmuxp load $CATKIN_WS_/src/kimera_multi/examples/example_real.yaml
fi
