'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2024-09-28 15:57:06
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-04-16 10:41:02
FilePath: /src/kimera_multi/examples/evo_offline.py
Version:
Description: Offline version of evo_real_time.py for batch processing of pose files

'''

'''
python evo_offline.py --date 12_08 --robot_num 3 --file_range 50-200 --file_step 10
'''

import matplotlib.pyplot as plt
import os
import glob
import copy
import sys
import pandas as pd
import numpy as np
from evo.core import metrics
from evo.tools import file_interface
from evo.core import sync
from evo.tools import plot
from evo.tools.settings import SETTINGS
import argparse
from tqdm import tqdm
import re

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
        description='Process and visualize trajectory data in offline mode')
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
    parser.add_argument('--log_dir', type=str, default='',
                        help='Custom log directory path (optional)')
    parser.add_argument('--gt_dir', type=str, default='',
                        help='Custom ground truth directory path (optional)')
    parser.add_argument('--output_dir', type=str, default='',
                        help='Custom output directory path (optional)')
    parser.add_argument('--file_range', type=str, default='all',
                        help='Range of files to process (e.g., "1-100", "50,100,150", or "all")')
    parser.add_argument('--file_step', type=int, default=5,
                        help='Step size between files when processing a range')
    return parser.parse_args()


# 设置目录和前缀
DIR_PREFIX = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets'


def setup_paths(date, custom_log_dir='', custom_gt_dir='', custom_output_dir=''):
    # Default paths based on date
    default_paths = {
        'LOG_DIR': f'{DIR_PREFIX}/{DATE2DATASET[date]}/log_data_{date}/',
        'GT_DIR': f'{DIR_PREFIX}/Kimera-Multi-Public-Data/ground_truth/{date[:2]}{date[3:]}/',
        'OUTPUT_DIR': f'{DIR_PREFIX}/paper/{DATE2DATASET[date]}/exp_range/'
    }

    # Override with custom paths if provided
    return {k: custom_path if custom_path else default_paths[k]
            for k, custom_path in zip(
                ['LOG_DIR', 'GT_DIR', 'OUTPUT_DIR'],
                [custom_log_dir, custom_gt_dir, custom_output_dir]
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


def plot_global_trajectory(all_gt_trajectories, all_est_trajectories, robot_names, save_path, file_num=None):
    """Plot all robot trajectories in a single global view"""
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    title = "Global Trajectory View"
    if file_num is not None:
        title += f" - File {file_num}"
    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # Compute global transformation
    global_transformation = compute_global_transformation(
        all_gt_trajectories, all_est_trajectories)

    # Plot ground truth trajectories
    for i, (gt_traj, robot_name) in enumerate(zip(all_gt_trajectories, robot_names)):
        if gt_traj is None:
            continue

        if i == len(robot_names) - 1:
            plot.traj(ax, plot.PlotMode.xy, gt_traj,
                      label="GT",
                      color='k',
                      plot_start_end_markers=True)
        else:
            # Plot ground truth as dotted line
            plot.traj(ax, plot.PlotMode.xy, gt_traj,
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


def plot_ape_metrics(ape_dict, robot_names, save_path):
    """Plot APE metrics for all robots"""
    fig = plt.figure(figsize=(15, 10))
    for num, robot_name in enumerate(robot_names):
        if robot_name not in ape_dict or ape_dict[robot_name].empty:
            continue

        ax = plt.subplot(2, 3, num + 1)
        ax2 = ax.twinx()

        l1, = ax.plot(ape_dict[robot_name]['ts'],
                      ape_dict[robot_name]['trans'],
                      label='translation', color='blue')
        l2, = ax.plot(ape_dict[robot_name]['ts'],
                      ape_dict[robot_name]['full'],
                      label='full', color='green')
        l3, = ax2.plot(ape_dict[robot_name]['ts'],
                       ape_dict[robot_name]['length'],
                       label='length', color='red', linestyle='--')

        ax.set_xlabel('File Number')
        ax.set_ylabel('APE (m)')
        ax2.set_ylabel('Length of Trajectory (m)', color='red')
        ax.set_title(f"{robot_name} APE")

        lines = [l1, l2, l3]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_global_metrics(global_metrics_dict, save_path):
    """Plot global metrics"""
    if global_metrics_dict.empty:
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    l1, = ax.plot(global_metrics_dict['ts'],
                  global_metrics_dict['trans'],
                  label='Global Translation Error', color='blue', linewidth=2)
    l2, = ax.plot(global_metrics_dict['ts'],
                  global_metrics_dict['full'],
                  label='Global Full Error', color='green', linewidth=2)
    l3, = ax2.plot(global_metrics_dict['ts'],
                   global_metrics_dict['length'],
                   label='Global Path Length', color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('File Number', fontsize=12)
    ax.set_ylabel('Global APE (m)', fontsize=12)
    ax2.set_ylabel('Global Path Length (m)', color='red', fontsize=12)
    ax.set_title('Global Trajectory Metrics', fontsize=14)

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='upper left', fontsize=10)

    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_trajectory_for_file(ad_traj_by_label, robot_names, save_path, file_num):
    """Plot trajectory for a specific file number"""
    fig = plt.figure(figsize=(15, 10))

    for num, robot_name in enumerate(robot_names):
        if num >= len(ad_traj_by_label) or "est" not in ad_traj_by_label[num] or "ref" not in ad_traj_by_label[num]:
            continue

        ax = plt.subplot(2, 3, num + 1)
        ax.set_title(f"{robot_name} Trajectory - File {file_num}")

        plot.traj(ax, plot.PlotMode.xy,
                  ad_traj_by_label[num]['est_aligned'],
                  label='est', color='blue',
                  plot_start_end_markers=True)
        plot.traj(ax, plot.PlotMode.xy,
                  ad_traj_by_label[num]['ref'],
                  label='ref', color='green',
                  plot_start_end_markers=True)
        plot.draw_correspondence_edges(ax,
                                       ad_traj_by_label[num]['est_aligned'],
                                       ad_traj_by_label[num]['ref'],
                                       plot.PlotMode.xy, alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def parse_file_range(range_str, max_file_num):
    """Parse file range string into list of file numbers"""
    if range_str == 'all':
        return list(range(1, max_file_num + 1))

    file_nums = []

    # Handle comma-separated values
    if ',' in range_str:
        parts = range_str.split(',')
        for part in parts:
            try:
                num = int(part.strip())
                if 1 <= num <= max_file_num:
                    file_nums.append(num)
            except ValueError:
                pass
        return sorted(file_nums)

    # Handle ranges like "1-100"
    if '-' in range_str:
        try:
            start, end = map(int, range_str.split('-'))
            start = max(1, start)
            end = min(max_file_num, end)
            return list(range(start, end + 1))
        except ValueError:
            pass

    # Handle single number
    try:
        num = int(range_str)
        if 1 <= num <= max_file_num:
            return [num]
    except ValueError:
        pass

    # Default to all files if parsing fails
    return list(range(1, max_file_num + 1))


def get_max_file_num(log_dir, robot_names, flag_multi, file_prefix='kimera_distributed_poses_tum_'):
    """Get the maximum file number across all robots"""
    max_num = 0

    if not flag_multi:
        return 1  # For single mode, there's only one file per robot

    for robot_name in robot_names:
        robot_dir = os.path.join(log_dir, robot_name, 'distributed/')
        if not os.path.exists(robot_dir):
            continue

        files = glob.glob(os.path.join(robot_dir, f'{file_prefix}*.tum'))
        for file_path in files:
            try:
                # Extract file number from filename
                file_num = int(re.search(r'(\d+)\.tum$', file_path).group(1))
                max_num = max(max_num, file_num)
            except (AttributeError, ValueError):
                continue

    return max_num


def process_file(file_num, traj_ref, log_dir, robot_names, flag_multi, file_prefix='kimera_distributed_poses_tum_'):
    """Process a single file number across all robots"""
    all_gt_trajectories = []
    all_est_trajectories = []
    ad_traj_by_label = [{} for _ in range(len(robot_names))]

    for num, robot_name in enumerate(robot_names):
        if num >= len(traj_ref) or traj_ref[num] is None:
            all_gt_trajectories.append(None)
            all_est_trajectories.append(None)
            continue

        if flag_multi:
            # For multi-robot mode
            robot_dir = os.path.join(log_dir, robot_name, 'distributed/')
            file_path = os.path.join(robot_dir, f'{file_prefix}{file_num}.tum')

            if not os.path.exists(file_path):
                all_gt_trajectories.append(None)
                all_est_trajectories.append(None)
                continue

            try:
                traj_est = file_interface.read_tum_trajectory_file(file_path)
                traj_ref_, traj_est = sync.associate_trajectories(
                    traj_ref[num], traj_est, max_diff=0.01)

                ad_traj_by_label[num] = {
                    "est": traj_est, "ref": traj_ref_
                }

                all_gt_trajectories.append(traj_ref_)
                all_est_trajectories.append(traj_est)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                all_gt_trajectories.append(None)
                all_est_trajectories.append(None)

        else:
            # For single-robot mode
            robot_dir = os.path.join(log_dir, robot_name, 'single/')
            file_path = os.path.join(robot_dir, 'traj_pgo.tum')

            if not os.path.exists(file_path) or not os.path.getsize(file_path):
                all_gt_trajectories.append(None)
                all_est_trajectories.append(None)
                continue

            try:
                # Check if file has enough content
                with open(file_path) as f:
                    if sum(1 for _ in f) < 100:
                        all_gt_trajectories.append(None)
                        all_est_trajectories.append(None)
                        continue

                traj_est = file_interface.read_tum_trajectory_file(file_path)
                traj_ref_, traj_est = sync.associate_trajectories(
                    traj_ref[num], traj_est, max_diff=0.01)

                ad_traj_by_label[num] = {
                    "est": traj_est, "ref": traj_ref_
                }

                all_gt_trajectories.append(traj_ref_)
                all_est_trajectories.append(traj_est)

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                all_gt_trajectories.append(None)
                all_est_trajectories.append(None)

    # Compute global transformation if in multi-robot mode
    global_transformation = None
    if flag_multi:
        global_transformation = compute_global_transformation(
            all_gt_trajectories, all_est_trajectories)

    # Process each robot with the same global transformation
    ape_results = {}
    for num, robot_name in enumerate(robot_names):
        if num >= len(ad_traj_by_label) or "est" not in ad_traj_by_label[num] or "ref" not in ad_traj_by_label[num]:
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

        # Calculate APE
        ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
        ape_full = metrics.APE(metrics.PoseRelation.full_transformation)

        ape_trans.process_data((traj_ref_, traj_est_aligned))
        ape_full.process_data((traj_ref_, traj_est_aligned))

        ape_results[robot_name] = {
            'ts': file_num,
            'length': traj_ref_.path_length,
            'trans': ape_trans.get_statistic(metrics.StatisticsType.rmse),
            'full': ape_full.get_statistic(metrics.StatisticsType.rmse)
        }

    # Calculate global metrics
    global_metrics = None
    if flag_multi:
        global_metrics_result, global_path_length = compute_global_metrics(
            all_gt_trajectories, all_est_trajectories)
        if global_metrics_result:
            global_metrics = {
                'ts': file_num,
                'length': global_path_length,
                'trans': global_metrics_result['trans'],
                'full': global_metrics_result['full']
            }

    return ape_results, global_metrics, ad_traj_by_label, all_gt_trajectories, all_est_trajectories


def main():
    args = parse_args()
    paths = setup_paths(args.date, args.log_dir, args.gt_dir, args.output_dir)

    log_dir = paths['LOG_DIR']
    gt_dir = paths['GT_DIR']
    output_dir = paths['OUTPUT_DIR']
    robot_num = args.robot_num
    flag_multi = args.flag_multi
    dataset_name = args.dataset
    file_step = args.file_step

    # Adjust directory for single mode
    if not flag_multi:
        output_dir = '/'.join(output_dir.split('/')[:-2]) + '/test_single/'
        print('Single mode')
    else:
        print('Multi mode')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Set file prefix
    file_prefix = 'kimera_distributed_poses_tum_'
    robot_names = ['acl_jackal', 'acl_jackal2', 'sparkal1',
                   'sparkal2', 'hathor', 'thoth'][:robot_num]

    # Load ground truth trajectories
    traj_ref = []
    print("Loading ground truth trajectories...")
    for num in range(robot_num):
        if dataset_name == 'Kimera-Multi':
            ref_file = os.path.join(
                gt_dir, f'modified_{robot_names[num]}_gt_odom.tum')
        elif dataset_name == 'EuRoC':
            ref_file = os.path.join(gt_dir, 'data.tum')
        else:
            raise ValueError("Invalid dataset name")

        try:
            traj = file_interface.read_tum_trajectory_file(ref_file)
            traj_ref.append(traj)
            print(f"Loaded trajectory for {robot_names[num]}")
        except Exception as e:
            print(f"Error loading trajectory for {robot_names[num]}: {e}")
            traj_ref.append(None)

    # Initialize dictionaries for storing results
    ape_dict = {robot_name: pd.DataFrame(columns=['ts', 'length', 'trans', 'full'])
                for robot_name in robot_names}
    global_metrics_dict = pd.DataFrame(
        columns=['ts', 'length', 'trans', 'full'])

    # Find maximum file number
    max_file_num = get_max_file_num(
        log_dir, robot_names, flag_multi, file_prefix)
    print(f"Found maximum file number: {max_file_num}")

    # Parse file range
    file_range = parse_file_range(args.file_range, max_file_num)
    if file_step > 1:
        file_range = [n for i, n in enumerate(
            file_range) if i % file_step == 0]
    if file_range[-1] != max_file_num:  # Always include the last file
        file_range.append(max_file_num)
    print(
        f"Processing files: {file_range[0]}...{file_range[-1]} (total: {len(file_range)})")

    # Process files
    for file_num in tqdm(file_range, desc="Processing files"):
        ape_results, global_metrics, ad_traj_by_label, all_gt_trajectories, all_est_trajectories = process_file(
            file_num, traj_ref, log_dir, robot_names, flag_multi, file_prefix)

        # Update APE results
        for robot_name, result in ape_results.items():
            new_row = pd.DataFrame([result])
            ape_dict[robot_name] = pd.concat(
                [ape_dict[robot_name], new_row], ignore_index=True)

        # Update global metrics
        if global_metrics:
            new_global_row = pd.DataFrame([global_metrics])
            global_metrics_dict = pd.concat(
                [global_metrics_dict, new_global_row], ignore_index=True)

        # Generate visualizations for specific checkpoints or the last file
        if file_num == file_range[-1] or file_num % 50 == 0:
            # Plot trajectory for this file
            plot_trajectory_for_file(
                ad_traj_by_label, robot_names,
                os.path.join(output_dir, f'trajectory_file_{file_num}.jpg'),
                file_num
            )

            # Plot global trajectory if in multi-robot mode
            if flag_multi:
                plot_global_trajectory(
                    all_gt_trajectories, all_est_trajectories, robot_names,
                    os.path.join(
                        output_dir, f'global_trajectory_file_{file_num}.jpg'),
                    file_num
                )

    # Generate final visualizations
    # APE plots
    plot_ape_metrics(ape_dict, robot_names,
                     os.path.join(output_dir, 'ape.jpg'))

    # Global metrics plot
    if not global_metrics_dict.empty:
        plot_global_metrics(global_metrics_dict, os.path.join(
            output_dir, 'global_metrics.jpg'))

    # Save data to CSV files
    for robot_name in robot_names:
        if not ape_dict[robot_name].empty:
            ape_dict[robot_name].to_csv(os.path.join(
                output_dir, f'ape_{robot_name}.csv'), index=False)
            print(f'Saved APE data for {robot_name}')

    if not global_metrics_dict.empty:
        global_metrics_dict.to_csv(os.path.join(
            output_dir, 'global_metrics.csv'), index=False)
        print('Saved global metrics data')

    print(f"All results saved to {output_dir}")


if __name__ == '__main__':
    main()
