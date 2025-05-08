#!/bin/bash
###
 # Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
 # @Author: Ronghai He
 # @Date: 2024-10-31 12:32:52
 # @LastEditors: RonghaiHe hrhkjys@qq.com
 # @LastEditTime: 2025-04-18 02:00:25
 # @FilePath: /src/kimera_multi/examples/run_multi_times.sh
 # @Version: 
 # @Description: 
 # 
### 

declare -A TIME2DATASET
TIME2DATASET=(
    ["10_14"]="campus_outdoor_10_14"
    ["12_07"]="campus_tunnels_12_07"
    ["12_08"]="campus_hybrid_12_08"
)  

# ROBOT_KIND: 1: single robot, 2: multi robot
ROBOT_KIND=${1:-"2"}
DATE=${2:-"12_07"}

DIR_BASE="/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/${TIME2DATASET[$DATE]}"

# Set directories based on robot kind
if [ $ROBOT_KIND = "1" ]; then
    # ./run_multi_times.sh 1 12_08
    echo "Running single robot"
    TARGET_SCRIPT="run.sh 4 $DATE"
    DIR_ORIGIN="${DIR_BASE}/exp_distributed"
else
    # ./run_multi_times.sh 2 12_08
    echo "Running multi robot"
    TARGET_SCRIPT="run.sh 3 $DATE"
    DIR_ORIGIN="${DIR_BASE}/exp_range"
fi

DIR_DEST="${DIR_ORIGIN}_/"

# Loop to run the target script multiple times
for i in {1..3}; do
    echo "Running iteration $i"
    gnome-terminal --title="run" -- /bin/bash -c "cd /media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/examples; bash $TARGET_SCRIPT"
    
    # Wait for processing to complete
    sleep 600
    
    # Check for Python script completion and cleanup
    while pgrep -f "evo_real_time.py" > /dev/null; do
        # echo "Waiting for processing to complete..."
        sleep 4
    done
    
    # Cleanup and prepare for next iteration
    tmux kill-server
    mv "/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/${TIME2DATASET[$DATE]}/log_data_${DATE}" "${DIR_ORIGIN}"
    mv "$DIR_ORIGIN" "${DIR_ORIGIN}${i}"
    mv "${DIR_ORIGIN}${i}" "$DIR_DEST"
    
    echo "Iteration $i completed"
done

echo "All iterations completed. Results are in ${DIR_DEST}"