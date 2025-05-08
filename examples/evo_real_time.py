'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2024-09-28 15:57:06
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-04-18 16:24:43
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
import numpy as np
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

# Colors for different trajectories
COLORS = {
    'gt': 'black',
    'distributed': 'blue',
    'single': 'red',
    'each': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2',
              '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', 'k']
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process and visualize trajectory data')
    parser.add_argument('--date', type=str, default='12_08',
                        choices=list(DATE2DATASET.keys()),
                        help='Date of the dataset (e.g., 12_07, 10_14, 12_08)')
    parser.add_argument('--robot_num', type=int, default=3,
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
    # Default paths based on date
    default_paths = {
        'LOG_DIR': f'{DIR_PREFIX}/{DATE2DATASET[date]}/log_data_{date}/',
        'GT_DIR': f'{DIR_PREFIX}/Kimera-Multi-Public-Data/ground_truth/{date[:2]}{date[3:]}/',
        'APE_DIR': f'{DIR_PREFIX}/paper/{DATE2DATASET[date]}/exp_range/'
    }

    # Override with custom paths if provided (using a more concise approach)
    return {k: custom_path if custom_path else default_paths[k]
            for k, custom_path in zip(
                ['LOG_DIR', 'GT_DIR', 'APE_DIR'],
                [custom_log_dir, custom_gt_dir, custom_ape_dir]
    )}


# Helper functions for global trajectory visualization
def compute_global_transformation(all_gt_trajectories, all_est_trajectories):
    """Compute the global transformation for aligning all trajectories"""
    # Filter out None trajectories
    valid_pairs = [(gt, est) for gt, est in zip(all_gt_trajectories, all_est_trajectories)
                   if gt is not None and est is not None]

    if not valid_pairs:
        return None

    # Merge all trajectories
    merged_gt = merge_trajectories([gt for gt, _ in valid_pairs])
    merged_est = merge_trajectories([est for _, est in valid_pairs])

    if merged_gt is None or merged_est is None:
        return None

    try:
        # Associate and align trajectories
        aligned_gt, aligned_est = sync.associate_trajectories(
            merged_gt, merged_est, max_diff=0.01)
        global_transform_r, global_transform_t, _ = aligned_est.align(
            aligned_gt, correct_scale=False)
        return {
            'rotation': global_transform_r,
            'translation': global_transform_t
        }
    except Exception as e:
        print(f"Error computing global transformation: {e}")
        return None


def compute_global_metrics(all_gt_trajectories, all_est_trajectories):
    """Compute ATE and APE metrics using global transformation"""
    # Get global transformation first
    global_transform = compute_global_transformation(
        all_gt_trajectories, all_est_trajectories)
    if global_transform is None:
        return None, 0

    # Calculate metrics for each robot using the same global transformation
    total_path_length = 0
    all_errors_trans = []
    all_errors_full = []

    for gt_traj, est_traj in zip(all_gt_trajectories, all_est_trajectories):
        if gt_traj is None or est_traj is None:
            continue

        try:
            # Associate trajectories
            aligned_gt, aligned_est = sync.associate_trajectories(
                gt_traj, est_traj, max_diff=0.01)

            # Update total path length
            total_path_length += aligned_gt.path_length

            # Apply the global transformation to this trajectory
            aligned_est_transformed = apply_transformation(
                aligned_est, global_transform)

            # Calculate metrics
            ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
            ape_full = metrics.APE(metrics.PoseRelation.full_transformation)

            ape_trans.process_data((aligned_gt, aligned_est_transformed))
            ape_full.process_data((aligned_gt, aligned_est_transformed))

            all_errors_trans.append(
                ape_trans.get_statistic(metrics.StatisticsType.rmse))
            all_errors_full.append(
                ape_full.get_statistic(metrics.StatisticsType.rmse))

        except Exception as e:
            print(f"Error computing metrics for trajectory: {e}")

    if not all_errors_trans or not all_errors_full:
        return None, 0

    # Calculate the average error across all trajectories
    avg_trans_error = sum(all_errors_trans) / len(all_errors_trans)
    avg_full_error = sum(all_errors_full) / len(all_errors_full)

    return {'trans': avg_trans_error, 'full': avg_full_error}, total_path_length


def apply_transformation(est_traj, transformation):
    """Apply a pre-computed transformation to an estimated trajectory"""
    if transformation is None or est_traj is None:
        return est_traj

    # Create copies of trajectory data
    positions = est_traj.positions_xyz.copy()
    orientations = est_traj.orientations_quat_wxyz.copy()
    timestamps = est_traj.timestamps.copy()

    # Apply the transformation
    r = transformation['rotation']
    t = transformation['translation']

    # Apply to all positions
    transformed_positions = np.array([r.dot(pos) + t for pos in positions])

    # Create a new trajectory object
    from evo.core.trajectory import PoseTrajectory3D
    transformed_traj = PoseTrajectory3D(
        positions_xyz=transformed_positions,
        orientations_quat_wxyz=orientations,
        timestamps=timestamps
    )

    return transformed_traj


def merge_trajectories(trajectories):
    """Merge multiple trajectories into one by concatenating their data"""
    if not trajectories:
        return None

    from evo.core.trajectory import merge
    try:
        # Filter out None trajectories
        valid_trajectories = [t for t in trajectories if t is not None]
        if not valid_trajectories:
            return None

        # Use the built-in merge function from evo
        merged_traj = merge(valid_trajectories)
        return merged_traj
    except Exception as e:
        print(f"Error merging trajectories: {e}")
        return None


def plot_global_trajectory(all_gt_trajectories, all_est_trajectories, robot_names, save_path):
    """Plot all robot trajectories in a single global view"""
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title("Global Trajectory View")
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # Compute global transformation
    global_transformation = compute_global_transformation(
        all_gt_trajectories, all_est_trajectories)

    # Plot ground truth trajectories
    for i, (gt_traj, robot_name) in enumerate(zip(all_gt_trajectories, robot_names)):
        if gt_traj is None:
            continue

        # Use consistent color scheme
        # color = COLORS['each'][i % len(COLORS['each'])]

        if i == len(robot_name) - 1:
            plot.traj(ax, plot.PlotMode.xy, gt_traj,
                      label=f"GT",
                      color='k',
                      plot_start_end_markers=True)
        else:
            # Plot ground truth as dotted line
            plot.traj(ax, plot.PlotMode.xy, gt_traj,
                      #   label=f"{robot_name} (GT)",
                      color='k',
                      plot_start_end_markers=True)

    # Plot estimated trajectories after applying global transformation
    for i, (est_traj, robot_name) in enumerate(zip(all_est_trajectories, robot_names)):
        if est_traj is None:
            continue

        # Apply global transformation if available
        if global_transformation is not None:
            est_traj = apply_transformation(est_traj, global_transformation)

        # Use consistent color scheme
        color = COLORS['each'][i % len(COLORS['each'])]

        # Plot estimated trajectory as solid line
        plot.traj(ax, plot.PlotMode.xy, est_traj,
                  label=f"{robot_name} (Est)",
                  color=color,
                  plot_start_end_markers=True)

    # Add legend and grid
    ax.legend()
    ax.grid(True)

    # Save figure
    plt.savefig(save_path)
    plt.close(fig)


# Initialize global variables
PREFIX = 'kimera_distributed_poses_tum_'
INTERVAL = 5
ROBOT_NAMES = ['acl_jackal', 'apis',
               'sobek', 'apis', 'hathor', 'thoth']
LOG_DIR = ""
GT_DIR = ""
APE_DIR = ""
ROBOT_NUM = 6
flag_multi = 1

# Initialize APE metrics with class instances instead of class references
ape_trans = None
ape_full = None
max_diff = 0.01

# Add new global variable for tracking global metrics
global_metrics_dict = pd.DataFrame(columns=['ts', 'length', 'trans', 'full'])

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

    # Save global metrics data
    if not global_metrics_dict.empty:
        global_metrics_dict.to_csv(os.path.join(
            APE_DIR, 'global_metrics.csv'), index=False)
        print('Saved global metrics data')

    # Save pose files
    save_pose_files()
    sys.exit(0)


# 捕获 SIGTERM 和 SIGHUP 信号
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGHUP, signal_handler)

# Add a function to count DPGO log files


def count_dpgo_logs(robot_dir):
    """Count the number of DPGO log files in the directory"""
    dpgo_files = glob.glob(os.path.join(robot_dir, 'dpgo_log_*.csv'))
    return len(dpgo_files)


newest_file_num = 0


def main(retry_count=10):
    args = parse_args()
    paths = setup_paths(args.date, args.log_dir, args.gt_dir, args.ape_dir)

    global LOG_DIR, GT_DIR, APE_DIR, ROBOT_NUM, ROBOT_NAMES, flag_multi, ape_trans, ape_full, newest_file_num, global_metrics_dict
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

    # Define these variables that were missing before
    TYPE_DIR = 'distributed/'
    JUDGE_IF_KILLED = 10000
    newest_file = None
    if not flag_multi:
        TYPE_DIR = 'single/'
        JUDGE_IF_KILLED = 2000

    attempt = 0
    ad_traj_by_label = [{} for _ in range(ROBOT_NUM)]

    # Add variable to track when to exit based on DPGO files
    dpgo_file_threshold = 33  # Adjust this threshold as needed

    while attempt < retry_count:
        try:
            while True:
                start_time = time.time()
                plt.close('all')

                # Check if we should terminate based on DPGO log count
                if flag_multi and ROBOT_NUM > 0:
                    first_robot_dir = os.path.join(
                        LOG_DIR, ROBOT_NAMES[0], 'distributed/')
                    dpgo_count = count_dpgo_logs(first_robot_dir)
                    if dpgo_count >= dpgo_file_threshold:
                        print(
                            f"Detected {dpgo_count} DPGO log files, exceeding threshold of {dpgo_file_threshold}. Terminating.")
                        # Save final results before exiting
                        for num in range(ROBOT_NUM):
                            ape_dict[ROBOT_NAMES[num]].to_csv(os.path.join(
                                APE_DIR, f'ape_{ROBOT_NAMES[num]}.csv'), index=False)
                        if not global_metrics_dict.empty:
                            global_metrics_dict.to_csv(os.path.join(
                                APE_DIR, 'global_metrics.csv'), index=False)
                        save_pose_files()
                        return

                # Collect all valid trajectories first to compute global transformation
                all_gt_trajectories = []
                all_est_trajectories = []
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
                        # for file in files[1:-1]:
                        #     # print(f'Removing {file}')
                        #     os.remove(file)

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

                    # Store original trajectories without individual alignment
                    ad_traj_by_label[num] = {
                        "est": traj_est, "ref": traj_ref_
                    }

                    all_gt_trajectories.append(traj_ref_)
                    all_est_trajectories.append(traj_est)

                global_transformation = None
                if flag_multi == 1:
                    # Compute global transformation once
                    global_transformation = compute_global_transformation(
                        all_gt_trajectories, all_est_trajectories)

                # Now process each robot with the same global transformation
                for num in range(ROBOT_NUM):
                    if "est" not in ad_traj_by_label[num] or "ref" not in ad_traj_by_label[num]:
                        continue

                    traj_ref_ = ad_traj_by_label[num]["ref"]
                    traj_est = ad_traj_by_label[num]["est"]

                    # Apply global transformation instead of individual alignment
                    if global_transformation is not None:
                        traj_est_aligned = apply_transformation(
                            traj_est, global_transformation)
                    else:
                        # Fall back to individual alignment if global fails
                        traj_est_aligned = copy.deepcopy(traj_est)
                        traj_est_aligned.align(
                            traj_ref_, correct_scale=False, correct_only_scale=False)

                    # Update with aligned trajectory
                    ad_traj_by_label[num]["est_aligned"] = traj_est_aligned

                    # 计算APE
                    data = (traj_ref_, traj_est_aligned)
                    ape_trans.process_data(data)
                    ape_full.process_data(data)

                    new_row = pd.DataFrame([{'ts': newest_file_num, 'length': traj_ref_.path_length, 'trans': ape_trans.get_statistic(
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
                                  ad_traj_by_label[num]['est_aligned'],
                                  label='est', color='blue',
                                  plot_start_end_markers=True)
                        plot.traj(ax_traj, plot.PlotMode.xy,
                                  ad_traj_by_label[num]['ref'],
                                  label='ref', color='green',
                                  plot_start_end_markers=True)
                        plot.draw_correspondence_edges(ax_traj,
                                                       ad_traj_by_label[num]['est_aligned'],
                                                       ad_traj_by_label[num]['ref'],
                                                       plot.PlotMode.xy, alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(APE_DIR, 'trajectory.jpg'))
                plt.close(fig_traj)

                if flag_multi:
                    # NEW: Global trajectory plot
                    # Collect all valid trajectories
                    all_gt_trajectories = []
                    all_est_trajectories = []
                    for num in range(ROBOT_NUM):
                        if "est" in ad_traj_by_label[num] and "ref" in ad_traj_by_label[num]:
                            all_gt_trajectories.append(
                                ad_traj_by_label[num]['ref'])
                            all_est_trajectories.append(
                                ad_traj_by_label[num]['est_aligned'])
                        else:
                            all_gt_trajectories.append(None)
                            all_est_trajectories.append(None)

                    # Calculate and store global metrics
                    global_metrics, global_path_length = compute_global_metrics(
                        all_gt_trajectories, all_est_trajectories)
                    if global_metrics:
                        new_global_row = pd.DataFrame([{
                            'ts': newest_file_num,
                            'length': global_path_length,
                            'trans': global_metrics['trans'],
                            'full': global_metrics['full']
                        }])
                        global_metrics_dict = pd.concat(
                            [global_metrics_dict, new_global_row], ignore_index=True)

                        # Save global metrics to file
                        global_metrics_dict.to_csv(os.path.join(
                            APE_DIR, 'global_metrics.csv'), index=False)

                    # NEW: Global metrics plot
                    if not global_metrics_dict.empty:
                        fig_global_metrics = plt.figure(figsize=(10, 8))
                        ax = fig_global_metrics.add_subplot(111)

                        # Create twin axis for path length
                        ax2 = ax.twinx()

                        # Plot global APE metrics
                        l1, = ax.plot(global_metrics_dict['ts'],
                                      global_metrics_dict['trans'],
                                      label='Global Translation Error', color='blue', linewidth=2)
                        l2, = ax.plot(global_metrics_dict['ts'],
                                      global_metrics_dict['full'],
                                      label='Global Full Error', color='green', linewidth=2)

                        # Plot global path length
                        l3, = ax2.plot(global_metrics_dict['ts'],
                                       global_metrics_dict['length'],
                                       label='Global Path Length', color='red', linestyle='--', linewidth=2)

                        # Set labels and title
                        ax.set_xlabel('Time', fontsize=12)
                        ax.set_ylabel('Global APE (m)', fontsize=12)
                        ax2.set_ylabel('Global Path Length (m)',
                                       color='red', fontsize=12)
                        ax.set_title(
                            'Global Trajectory Metrics Over Time', fontsize=14)

                        # Combine legends from both axes
                        lines = [l1, l2, l3]
                        labels = [l.get_label() for l in lines]
                        ax.legend(lines, labels, loc='upper left', fontsize=10)

                        ax.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(
                            APE_DIR, 'global_metrics.jpg'))
                        plt.close(fig_global_metrics)

                    # Plot global trajectory
                    if any(t is not None for t in all_est_trajectories):
                        plot_global_trajectory(all_gt_trajectories, all_est_trajectories,
                                               ROBOT_NAMES, os.path.join(APE_DIR, 'global_trajectory.jpg'))

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
            if attempt >= retry_count or str(e)[:6] == 'Killed':
                print("Max retry attempts or time reached. Exiting.")
                plt.close('all')
                for num in range(ROBOT_NUM):
                    ape_dict[ROBOT_NAMES[num]].to_csv(os.path.join(
                        APE_DIR, f'ape_{ROBOT_NAMES[num]}.csv'), index=False)
                    print(f'Saved APE data for {ROBOT_NAMES[num]}')
                # Also save global metrics on exit
                if not global_metrics_dict.empty:
                    global_metrics_dict.to_csv(os.path.join(
                        APE_DIR, 'global_metrics.csv'), index=False)
                    print('Saved global metrics data')
                break
            else:
                print(f'The {attempt}/{retry_count} Retrying...')
                time.sleep(5)


if __name__ == '__main__':
    main()
    # main(1)
