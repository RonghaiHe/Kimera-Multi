'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2024-09-28 15:57:06
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-03-17 20:46:55
FilePath: /src/kimera_multi/examples/evo_real_time.py
Version: 
Description: 

'''

import matplotlib.pyplot as plt
import signal
import os
import glob
import copy
import sys
import time
import pandas as pd
from evo.core import metrics
from evo.tools import file_interface
from evo.core import sync
from evo.tools import plot
from evo.tools.settings import SETTINGS
import argparse

SETTINGS.plot_usetex = False
plot.apply_settings(SETTINGS)

# Add dictionary for date to dataset mapping
DATE2DATASET = {
    '12_07': 'campus_tunnels_12_07',
    '10_14': 'campus_outdoor_10_14',
    '12_08': 'campus_hybrid_12_08'
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process and visualize trajectory data')
    parser.add_argument('--date', type=str, default='12_07',
                        choices=list(DATE2DATASET.keys()),
                        help='Date of the dataset (e.g., 12_07, 10_14, 12_08)')
    parser.add_argument('--robot_num', type=int, default=6,
                        help='Number of robots to process')
    parser.add_argument('--flag_multi', type=int, default=1,
                        choices=[1, 0],
                        help='Flag for multi or single robot')
    parser.add_argument('--dataset', type=str, default='Kimera-Multi',
                        help='Name of the dataset (optional)')
    # Add new arguments for directory paths
    parser.add_argument('--log_dir', type=str, default='',
                        help='Custom log directory path (optional)')
    parser.add_argument('--gt_dir', type=str, default='',
                        help='Custom ground truth directory path (optional)')
    parser.add_argument('--ape_dir', type=str, default='',
                        help='Custom APE output directory path (optional)')
    return parser.parse_args()


# 设置目录和前缀
DIR_PREFIX = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets'


def setup_paths(date, custom_log_dir='', custom_gt_dir='', custom_ape_dir=''):
    # Default paths
    default_paths = {
        'LOG_DIR': f'{DIR_PREFIX}/{DATE2DATASET[date]}/log_data_{date}/',
        'GT_DIR': f'{DIR_PREFIX}/Kimera-Multi-Public-Data/ground_truth/{date[:2]}{date[3:]}/',
        'APE_DIR': f'{DIR_PREFIX}/evo_try/{DATE2DATASET[date]}/test_distributed/'
    }

    # Override with custom paths if provided
    paths = {
        'LOG_DIR': custom_log_dir if custom_log_dir else default_paths['LOG_DIR'],
        'GT_DIR': custom_gt_dir if custom_gt_dir else default_paths['GT_DIR'],
        'APE_DIR': custom_ape_dir if custom_ape_dir else default_paths['APE_DIR']
    }

    return paths


# Initialize global variables
PREFIX = 'kimera_distributed_poses_tum_'
INTERVAL = 5
ROBOT_NAMES = ['acl_jackal', 'acl_jackal2',
               'sparkal1', 'sparkal2', 'hathor', 'thoth']
LOG_DIR = ""
GT_DIR = ""
APE_DIR = ""
ROBOT_NUM = 6
flag_multi = 1

# Initialize APE metrics with class instances instead of class references
ape_trans = None
ape_full = None
max_diff = 0.01

# 定义信号处理函数


def save_pose_files():
    """Save the latest pose files for all robots"""
    for num in range(ROBOT_NUM):
        if not flag_multi:
            src_file = os.path.join(
                LOG_DIR, ROBOT_NAMES[num], 'single/traj_pgo.tum')
            if os.path.exists(src_file):
                dst_file = os.path.join(
                    APE_DIR, f'final_pose_{ROBOT_NAMES[num]}.csv')
                # Convert TUM format to CSV
                data = pd.read_csv(src_file, sep=' ', header=None)
                data.columns = ['timestamp', 'x',
                                'y', 'z', 'qx', 'qy', 'qz', 'qw']
                data.to_csv(dst_file, index=False)
        else:
            robot_dir = os.path.join(LOG_DIR, ROBOT_NAMES[num], 'distributed/')
            pose_files = glob.glob(os.path.join(robot_dir, f'{PREFIX}*.tum'))
            if pose_files:
                latest_file = max(pose_files, key=os.path.getmtime)
                dst_file = os.path.join(
                    APE_DIR, f'final_pose_{ROBOT_NAMES[num]}.csv')
                # Convert TUM format to CSV
                data = pd.read_csv(latest_file, sep=' ', header=None)
                data.columns = ['timestamp', 'x',
                                'y', 'z', 'qx', 'qy', 'qz', 'qw']
                data.to_csv(dst_file, index=False)


def signal_handler(_sig, _frame):
    plt.close('all')
    if not os.path.exists(APE_DIR):
        os.makedirs(APE_DIR)

    # Save APE data
    for num in range(ROBOT_NUM):
        ape_dict[ROBOT_NAMES[num]].to_csv(os.path.join(
            APE_DIR, f'ape_{ROBOT_NAMES[num]}.csv'), index=False)
        print(f'Saved APE data for {ROBOT_NAMES[num]}')

    # Save pose files
    save_pose_files()
    sys.exit(0)


# 捕获 SIGTERM 和 SIGHUP 信号
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)

newest_file_num = 0


def main(retry_count=10):
    args = parse_args()
    paths = setup_paths(args.date, args.log_dir, args.gt_dir, args.ape_dir)

    global LOG_DIR, GT_DIR, APE_DIR, ROBOT_NUM, ROBOT_NAMES, flag_multi, ape_trans, ape_full, newest_file_num
    LOG_DIR = paths['LOG_DIR']
    GT_DIR = paths['GT_DIR']
    APE_DIR = paths['APE_DIR']
    ROBOT_NUM = args.robot_num

    # Adjust ROBOT_NAMES based on robot_num
    flag_multi = args.flag_multi
    dataset_name = args.dataset
    ROBOT_NAMES = ROBOT_NAMES[:ROBOT_NUM]

    # Initialize APE metrics inside main
    ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
    ape_full = metrics.APE(metrics.PoseRelation.full_transformation)

    # Initialize traj_ref properly as a list
    traj_ref = []
    print("Loading ground truth trajectories...")
    for num in range(ROBOT_NUM):
        if dataset_name == 'Kimera-Multi':
            ref_file = os.path.join(
                GT_DIR, f'modified_{ROBOT_NAMES[num]}_gt_odom.tum')
        elif dataset_name == 'EuRoC':
            ref_file = os.path.join(GT_DIR, 'data.tum')
        else:
            raise ValueError("Invalid dataset name")
        try:
            traj = file_interface.read_tum_trajectory_file(ref_file)
            traj_ref.append(traj)
            print(f"Loaded trajectory for {ROBOT_NAMES[num]}")
        except Exception as e:
            print(f"Error loading trajectory for {ROBOT_NAMES[num]}: {e}")
            traj_ref.append(None)

    # Initialize ape_dict after ROBOT_NAMES is properly set
    global ape_dict
    ape_dict = {ROBOT_NAMES[num]: pd.DataFrame(columns=['ts', 'length', 'trans', 'full'])
                for num in range(ROBOT_NUM)}

    if not flag_multi:
        # modify the last directory to be single
        APE_DIR = '/'.join(APE_DIR.split('/')[:-2]) + '/test_single/'
        print('Single mode')
    else:
        print('Multi mode')

    if not os.path.exists(APE_DIR):
        os.makedirs(APE_DIR)

    TYPE_DIR = 'distributed/'
    JUDGE_IF_KILLED = 2000
    newest_file = None
    if not flag_multi:
        TYPE_DIR = 'single/'
        JUDGE_IF_KILLED = 2010
        # newest_file_num = 0

    attempt = 0
    ad_traj_by_label = [{} for _ in range(ROBOT_NUM)]
    while attempt < retry_count:

        try:
            while True:
                start_time = time.time()
                plt.close('all')
                for num in range(ROBOT_NUM):
                    LOG_DIR_ROBOT = os.path.join(
                        LOG_DIR, ROBOT_NAMES[num], TYPE_DIR)

                    if not flag_multi:
                        if (num == 0):
                            newest_file_num += INTERVAL
                        # if (newest_file_num < 30):
                        #     time.sleep(INTERVAL)
                        #     break
                        newest_file = os.path.join(
                            LOG_DIR_ROBOT, 'traj_pgo.tum')

                        # 如果文件不存在或文件是空的，跳过
                        if not os.path.exists(newest_file) or not os.path.getsize(newest_file):
                            continue

                        # 获取文件内容行数
                        with open(newest_file) as f:
                            row_num = sum(1 for _ in f)
                        if row_num < 100:
                            continue

                    else:
                        # 存储所有位姿文件
                        files = []

                        # 扫描目录
                        for file_path in glob.glob(os.path.join(LOG_DIR_ROBOT, f'{PREFIX}*.tum')):
                            files.append(file_path)

                        if len(files) < 1:
                            # print(f'Not enough files for {ROBOT_NAMES[num]}')
                            continue
                        newest_file = None

                        # 按最后修改时间排序
                        files.sort(key=lambda x: os.path.getmtime(x))

                        # 保留最新的和最旧的文件
                        newest_file = files[-1]

                        # 删除其他文件
                        for file in files[1:-1]:
                            # print(f'Removing {file}')
                            os.remove(file)

                        # 获取文件名称的数字部分
                        newest_file_num = int(
                            newest_file.split('_')[-1].split('.')[0])
                    if newest_file_num >= JUDGE_IF_KILLED:
                        raise ValueError(
                            f'Killed for {newest_file_num} >= {JUDGE_IF_KILLED}')
                    print(
                        f'Processing {newest_file_num} of {ROBOT_NAMES[num]}')
                    # 获取文件内容行数
                    if not flag_multi:
                        print(
                            f'File of {ROBOT_NAMES[num]} has {row_num} lines')

                    traj_est = file_interface.read_tum_trajectory_file(
                        newest_file)
                    traj_ref_, traj_est = sync.associate_trajectories(
                        traj_ref[num], traj_est, max_diff)

                    traj_est_aligned = copy.deepcopy(traj_est)
                    traj_est_aligned.align(
                        traj_ref_, correct_scale=False, correct_only_scale=False)

                    ad_traj_by_label[num] = {
                        "est": traj_est_aligned, "ref": traj_ref_}

                    # 计算APE
                    data = (traj_ref_, traj_est_aligned)
                    ape_trans.process_data(data)
                    ape_full.process_data(data)

                    # new_row = pd.DataFrame([{'ts': newest_file_num, 'count': len(traj_est_aligned.timestamps), 'trans': ape_trans.get_statistic(
                    #     metrics.StatisticsType.rmse), 'full': ape_full.get_statistic(metrics.StatisticsType.rmse)}])
                    new_row = pd.DataFrame([{'ts': newest_file_num, 'length': traj_est_aligned.path_length, 'trans': ape_trans.get_statistic(
                        metrics.StatisticsType.rmse), 'full': ape_full.get_statistic(metrics.StatisticsType.rmse)}])
                    ape_dict[ROBOT_NAMES[num]] = pd.concat(
                        [ape_dict[ROBOT_NAMES[num]], new_row], ignore_index=True)

                # Create separate figures for APE and trajectory plots
                plt.switch_backend('Agg')  # Use a non-interactive backend

                # APE Plot
                fig_ape = plt.figure(figsize=(15, 10))
                for num in range(ROBOT_NUM):
                    if "est" in ad_traj_by_label[num]:
                        ax = plt.subplot(2, 3, num+1)

                        # Create twin axis for pose count
                        ax2 = ax.twinx()

                        # Plot APE metrics
                        l1, = ax.plot(ape_dict[ROBOT_NAMES[num]]['ts'],
                                      ape_dict[ROBOT_NAMES[num]]['trans'],
                                      label='translation', color='blue')
                        l2, = ax.plot(ape_dict[ROBOT_NAMES[num]]['ts'],
                                      ape_dict[ROBOT_NAMES[num]]['full'],
                                      label='full', color='green')

                        # Plot pose length
                        # l3, = ax2.plot(ape_dict[ROBOT_NAMES[num]]['ts'],
                        #                ape_dict[ROBOT_NAMES[num]]['count'],
                        #                label='poses', color='red', linestyle='--')
                        l3, = ax2.plot(ape_dict[ROBOT_NAMES[num]]['ts'],
                                       ape_dict[ROBOT_NAMES[num]]['length'],
                                       label='length', color='red', linestyle='--')

                        # Set labels and title
                        ax.set_xlabel('Time')
                        ax.set_ylabel('APE (m)')
                        ax2.set_ylabel('Length of Trajectory (m)', color='red')
                        ax.set_title(f"{ROBOT_NAMES[num]} APE")

                        # Combine legends from both axes
                        lines = [l1, l2, l3]
                        labels = [l.get_label() for l in lines]
                        ax.legend(lines, labels, loc='upper left')

                        ax.grid(True)

                plt.tight_layout()
                plt.savefig(os.path.join(APE_DIR, 'ape.jpg'))
                plt.close(fig_ape)

                # Trajectory Plot
                fig_traj = plt.figure(figsize=(15, 10))
                for num in range(ROBOT_NUM):
                    if "est" in ad_traj_by_label[num] and "ref" in ad_traj_by_label[num]:
                        ax_traj = plt.subplot(2, 3, num+1)
                        ax_traj.set_title(f"{ROBOT_NAMES[num]} Trajectory")
                        plot.traj(ax_traj, plot.PlotMode.xy,
                                  ad_traj_by_label[num]['est'],
                                  label='est', color='blue',
                                  plot_start_end_markers=True)
                        plot.traj(ax_traj, plot.PlotMode.xy,
                                  ad_traj_by_label[num]['ref'],
                                  label='ref', color='green',
                                  plot_start_end_markers=True)
                        plot.draw_correspondence_edges(ax_traj,
                                                       ad_traj_by_label[num]['est'],
                                                       ad_traj_by_label[num]['ref'],
                                                       plot.PlotMode.xy, alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(APE_DIR, 'trajectory.jpg'))
                plt.close(fig_traj)

                print('-'*10)

                # 计算剩余的睡眠时间
                elapsed_time = time.time() - start_time
                if elapsed_time < INTERVAL:
                    sleep_time = INTERVAL - elapsed_time
                    time.sleep(sleep_time)
                    print(f'Sleeping for {sleep_time} seconds')

        except Exception as e:
            # 输出错误原因和对应的行
            print(
                f'Exiting for {e}, which is in line {sys.exc_info()[-1].tb_lineno}')

            attempt += 1
            if attempt >= retry_count or e[:6] == 'Killed':
                print("Max retry attempts or time reached. Exiting.")
                plt.close('all')
                for num in range(ROBOT_NUM):
                    ape_dict[ROBOT_NAMES[num]].to_csv(os.path.join(
                        APE_DIR, f'ape_{ROBOT_NAMES[num]}.csv'), index=False)
                    print(f'Saved APE data for {ROBOT_NAMES[num]}')
                break
            else:
                print(f'The {attempt}/{retry_count} Retrying...')
                time.sleep(5)


if __name__ == '__main__':
    main()
    # main(1)
