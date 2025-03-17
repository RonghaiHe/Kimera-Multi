#usr/bin/sh
###
 # Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
 # @Author: Ronghai He
 # @Date: 2025-12-28 19:56:40
 # @LastEditors: RonghaiHe hrhkjys@qq.com
 # @LastEditTime: 2025-03-14 12:24:43
 # @FilePath: /src/kimera_multi/evaluation/extract_lc_images.sh
 # @Description: Run programs and move images
 # 
### 

DATE="1207"
CURRENT_DIR=$(dirname "$0")

# Check if ros-noetic-image-transport-plugins is installed
if ! dpkg -l | grep ros-${ROS_DISTRO}-image-transport-plugins; then
    sudo apt-get install -y ros-${ROS_DISTRO}-image-transport-plugins
fi

source /media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/devel/setup.bash

# Extract sub-rosbag for specific image and merge them into one image
gnome-terminal --title="run" -- /bin/bash -c "source ~/miniconda3/bin/activate env3_9; python ${CURRENT_DIR}/extract_lc_images.py --date $DATE --sta_basic_path /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_tunnels_12_07/test_distributed_mono_count_kf_undistort/test_distributed3"

# Run launch file to extract image from the topic
gnome-terminal --title="run" -- /bin/bash -c "roslaunch kimera_multi extract_lc_images.launch"

mkdir -p "./lc_images_$DATE"

while true; do
    sleep 1
    # Check for jpg files and move them safely
    if [ $(find "$HOME/.ros" -maxdepth 1 -name "*.jpg" | wc -l) -gt 0 ] ; then
        find "$HOME/.ros" -name "*.jpg" -exec mv {} "./lc_images_$DATE/temp_images/" \;
    fi
done