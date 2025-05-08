'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-03-01
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-04-18 22:28:07
FilePath: /src/kimera_multi/examples/plot_multi_trajectory.py
Description:
  Script to read multiple trajectory files and plot them for comparison.
  For each robot, plot its ground truth and multiple estimated trajectories.
'''
import scienceplots
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import glob
import argparse
import numpy as np
import pandas as pd
from evo.tools import file_interface
from evo.core import sync, metrics
from evo.tools import plot
from evo.tools.settings import SETTINGS

'''
python plot_multi_trajectory.py --robot_num 3 \
    --flag_multi 10 \
    --gt_dir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1208/ \
        --traj_dir1 /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_range_/exp_range1/log_data_12_08 \
        --traj_name1 Kimera

python plot_multi_trajectory.py --robot_num 3 \
    --flag_multi 1 \
    --gt_dir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1208/ \
        --traj_dir1 /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_distributed_/exp_distributed1/log_data_12_08 \
        --traj_name1 Kimera

python plot_multi_trajectory.py --robot_num 3 \
    --flag_multi 0 \
    --gt_dir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1207/ \
        --traj_dir1 /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_tunnels_12_07/exp_single_/exp_single1/log_data_12_07 \
        --traj_name1 Kimera        

python plot_multi_trajectory.py --robot_num 3 \
    --flag_multi 1 \
    --gt_dir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1207/ \
        --traj_dir1 /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_tunnels_12_07/exp_distributed_/exp_distributed1/log_data_12_07 \
        --traj_name1 Kimera-Multi

python plot_multi_trajectory.py --robot_num 3 \
    --flag_multi 1 \
    --gt_dir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1207/ \
        --traj_dir1 /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_tunnels_12_07/exp_range_/exp_range1/log_data_12_07 \
        --traj_name1 Ours          
'''

SETTINGS.plot_usetex = False  # True
SETTINGS.plot_linewidth = 2
plot.apply_settings(SETTINGS)

plt.style.use(['science', 'ieee', 'no-latex', 'cjk-sc-font'])
plt.rcParams.update({
    'font.family': 'Noto Serif CJK JP',
    'axes.unicode_minus': False,
})
mpl.rcParams['lines.linewidth'] = 2.0


# Robot names
ROBOT_NAMES = ['acl_jackal', 'acl_jackal2',
               'apis', 'sparkal2', 'hathor', 'thoth']

# Base directory for datasets
DIR_PREFIX = '/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets'

# Trajectory file prefixes
SINGLE_TRAJ_FILE = 'traj_pgo.tum'
DISTRIBUTED_PREFIX = 'kimera_distributed_poses_tum_'

# Colors for different trajectory types
#    blue,       orange,    green,     red,      purple,    brown,    pink,      gray,      yellow,     cyan
# ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLORS = {
    'gt': 'black',
    'distributed': 'blue',
    'single': 'red',
    'each': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2',
              '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', 'k']
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot multiple trajectories for comparison')
    parser.add_argument('--robot_num', type=int, default=3,
                        help='Number of robots to process')
    parser.add_argument('--dataset', type=str, default='Kimera-Multi',
                        help='Name of the dataset')
    parser.add_argument('--gt_dir', type=str, default="/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1207/",
                        help='Ground truth directory path')
    parser.add_argument('--output_dir', type=str, required=False, default='./traj',
                        help='Output directory for trajectory plots')
    parser.add_argument('--flag_multi', type=int, default=0,
                        choices=[0, 1],
                        help='Flag for multi or single robot mode (0=single, 1=multi)')

    # Make trajectory directories optional with a minimum of 1
    parser.add_argument('--traj_dir1', type=str, default="/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_tunnels_12_07/exp_single_/exp_single1/log_data_12_07",
                        help='Directory path for first set of trajectory files')
    parser.add_argument('--traj_dir2', type=str, required=False, default='',
                        help='Directory path for second set of trajectory files')
    parser.add_argument('--traj_dir3', type=str, required=False, default='',
                        help='Directory path for third set of trajectory files')
    parser.add_argument('--traj_dir4', type=str, required=False, default='',
                        help='Directory path for fourth set of trajectory files')

    # Add trajectory names for the legend
    parser.add_argument('--traj_name1', type=str, default='Method 1',
                        help='Name for first trajectory set')
    parser.add_argument('--traj_name2', type=str, default='Method 2',
                        help='Name for second trajectory set')
    parser.add_argument('--traj_name3', type=str, default='Method 3',
                        help='Name for third trajectory set')
    parser.add_argument('--traj_name4', type=str, default='Method 4',
                        help='Name for fourth trajectory set')

    # Add custom file structure support
    parser.add_argument('--traj_file_pattern', type=str, default='distributed/kimera_distributed_poses_tum_3180.tum',
                        help='Pattern for trajectory file relative to traj_dir/robot_name/')

    # Add downsampling parameter for plotting
    parser.add_argument('--downsample', type=int, default=4,
                        help='Downsample factor for trajectory plotting (1 = no downsampling)')

    return parser.parse_args()


def read_ground_truth(gt_dir, robot_names, dataset_name):
    """Read ground truth trajectories for all robots"""
    print("Loading ground truth trajectories...")
    gt_trajectories = []

    for robot_name in robot_names:
        if dataset_name == 'Kimera-Multi':
            ref_file = os.path.join(
                gt_dir, f'modified_{robot_name}_gt_odom.tum')
        elif dataset_name == 'EuRoC':
            ref_file = os.path.join(gt_dir, 'data.tum')
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        try:
            traj = file_interface.read_tum_trajectory_file(ref_file)
            gt_trajectories.append(traj)
            print(f"Loaded GT trajectory for {robot_name}")
        except Exception as e:
            print(f"Error loading GT trajectory for {robot_name}: {e}")
            gt_trajectories.append(None)

    return gt_trajectories


def read_robot_trajectory(traj_path):
    """Read a single trajectory file"""
    try:
        if os.path.exists(traj_path):
            traj = file_interface.read_tum_trajectory_file(traj_path)
            return traj
        else:
            print(f"File not found: {traj_path}")
            return None
    except Exception as e:
        print(f"Error reading trajectory {traj_path}: {e}")
        return None


def read_robot_trajectories(robot_name, traj_dirs, traj_names, file_pattern, flag_multi=1):
    """Read trajectories for a single robot from multiple directories"""
    trajectories = []

    for i, traj_dir in enumerate(traj_dirs):
        if not traj_dir:  # Skip empty directory arguments
            continue

        # Handle different file path structures based on flag_multi
        if flag_multi:
            # Multi-robot mode - use file pattern from args
            traj_path = os.path.join(traj_dir, robot_name, file_pattern)
        else:
            # Single-robot mode - use single/traj_pgo.tum
            traj_path = os.path.join(
                traj_dir, robot_name, 'single', SINGLE_TRAJ_FILE)

        traj = read_robot_trajectory(traj_path)
        if traj is not None:
            trajectories.append((traj_names[i], traj))
            print(
                f"Loaded trajectory {traj_names[i]} for {robot_name} from {traj_path}")

    return trajectories


def plot_robot_trajectories(gt_traj, robot_trajectories, robot_name, output_dir, downsample=1):
    """Create plot for a single robot showing GT and multiple estimated trajectories"""
    if gt_traj is None:
        print(f"Skipping plot for {robot_name}: no ground truth available")
        return

    if not robot_trajectories:
        print(f"Skipping plot for {robot_name}: no trajectories available")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(f"{robot_name}")
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # Plot ground truth with downsampling
    gt_downsampled = downsample_trajectory(gt_traj, downsample)
    plot.traj(ax, plot.PlotMode.xy, gt_downsampled, label='真实位置',
              color=COLORS['gt'], plot_start_end_markers=True)

    # Plot estimated trajectories with downsampling
    for i, (traj_name, traj_est) in enumerate(robot_trajectories):
        try:
            traj_ref, traj_est = sync.associate_trajectories(
                gt_traj, traj_est, max_diff=0.01)
            traj_est.align(traj_ref, correct_scale=False,
                           correct_only_scale=False)

            # Choose color from the each robots' colors list
            color_idx = i % len(COLORS['each'])
            color = COLORS['each'][color_idx]

            # Downsample for plotting
            traj_est_downsampled = downsample_trajectory(traj_est, downsample)

            plot.traj(ax, plot.PlotMode.xy, traj_est_downsampled, label=traj_name.replace('_', '-'),
                      color=color, plot_start_end_markers=True)
        except Exception as e:
            print(
                f"Error processing trajectory {traj_name} for {robot_name}: {e}")
            continue

    ax.legend()
    ax.grid(True)

    # Save the figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fig.savefig(os.path.join(
        output_dir, f'{robot_name}_trajectories.png'), dpi=300)  # , bbox_inches='tight')
    plt.close(fig)
    print(f"Saved trajectory plot for {robot_name}")


def compute_transformation(gt_traj, est_traj):
    """Compute the transformation between ground truth and estimated trajectory"""
    try:
        # For a single trajectory pair
        if hasattr(gt_traj, 'timestamps') and hasattr(est_traj, 'timestamps'):
            traj_ref, traj_est = sync.associate_trajectories(
                gt_traj, est_traj, max_diff=0.01)
            # Use trajectory's align method to compute the transformation
            r, t, s = traj_est.align(traj_ref, correct_scale=False)
            return {
                'rotation': r,
                'translation': t
            }
        else:
            # We're dealing with a list of trajectories, fall back to using the first one
            # that has a valid match
            for gt, est in zip(gt_traj, est_traj):
                if gt is None or est is None:
                    continue
                try:
                    # Try to compute with this pair
                    traj_ref, traj_est = sync.associate_trajectories(
                        gt, est, max_diff=0.01)
                    r, t, s = traj_est.align(traj_ref, correct_scale=False)
                    return {
                        'rotation': r,
                        'translation': t
                    }
                except Exception:
                    # Try the next pair
                    continue

            # If we get here, no valid transformation was found
            print("Could not compute a valid transformation from any trajectory pair")
            return None
    except Exception as e:
        print(f"Error computing transformation: {e}")
        return None


def apply_transformation(est_traj, transformation):
    """Apply a pre-computed transformation to an estimated trajectory"""
    if transformation is None:
        return est_traj

    # Instead of using est_traj.copy() which doesn't exist,
    # create a new trajectory object with the same data
    positions = est_traj.positions_xyz.copy()
    orientations = est_traj.orientations_quat_wxyz.copy()
    timestamps = est_traj.timestamps.copy()

    # Apply the transformation
    r = transformation['rotation']
    t = transformation['translation']

    # Apply to all positions
    transformed_positions = []
    for pos in positions:
        transformed_pos = r.dot(pos) + t
        transformed_positions.append(transformed_pos)

    transformed_positions = np.array(transformed_positions)

    # Create a new trajectory object
    from evo.core.trajectory import PoseTrajectory3D
    transformed_traj = PoseTrajectory3D(
        positions_xyz=transformed_positions,
        orientations_quat_wxyz=orientations,
        timestamps=timestamps
    )

    return transformed_traj


def merge_trajectories(trajectories):
    """
    Merge multiple trajectories into one by concatenating their data.
    """
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


def plot_all_robot_trajectories(all_gt_trajectories, all_robot_trajectories, robot_names, output_dir, ref_robot_name=None, ref_traj_name=None, downsample=1, data_name="", flag_multi=False):
    """Create a separate image for each trajectory type with all robots"""

    traj_names = []
    for robot_trajs in all_robot_trajectories:
        if robot_trajs:
            for traj_name, _ in robot_trajs:
                if traj_name not in traj_names:
                    traj_names.append(traj_name)

    # Limit to 4 trajectory types
    for idx, traj_name in enumerate(traj_names[:4]):
        # Create a new figure for each trajectory type
        fig, ax = plt.subplots(figsize=(10, 8))
        # ax.set_title(f"Trajectory: {traj_name}")
        ax.set_xlabel(r'$X$ [m]')
        ax.set_ylabel(r'$Y$ [m]')

        # For multi-robot mode, compute a single global transformation for this trajectory type
        global_transform = None
        if flag_multi:
            # Collect all ground truth and estimated trajectories for this method
            method_gt_trajectories = []
            method_est_trajectories = []

            # First associate each trajectory pair
            associated_gt_trajs = []
            associated_est_trajs = []

            for i, (robot_name, gt_traj) in enumerate(zip(robot_names, all_gt_trajectories)):
                if gt_traj is None:
                    continue

                # Find this trajectory type for the current robot
                est_traj = None
                for name, traj in all_robot_trajectories[i]:
                    if name == traj_name:
                        est_traj = traj
                        break

                if est_traj is None:
                    continue

                # Associate trajectories (but don't align yet)
                try:
                    traj_ref, traj_est = sync.associate_trajectories(
                        gt_traj, est_traj, max_diff=0.01)

                    # Store the associated trajectories
                    associated_gt_trajs.append(traj_ref)
                    associated_est_trajs.append(traj_est)

                except Exception as e:
                    print(
                        f"Error associating trajectory for {robot_name}: {e}")
                    continue

            # Merge all associated trajectories for this method
            if associated_gt_trajs and associated_est_trajs:
                merged_gt = merge_trajectories(associated_gt_trajs)
                merged_est = merge_trajectories(associated_est_trajs)

                if merged_gt is not None and merged_est is not None:
                    # Compute a single global transformation for this method
                    global_transform = compute_transformation(
                        merged_gt, merged_est)
                    print(f"Using global transformation for {traj_name}")

        # Process each robot
        for i, (robot_name, gt_traj) in enumerate(zip(robot_names, all_gt_trajectories)):
            if gt_traj is None:
                continue

            # Find this trajectory type for the current robot
            current_traj = None
            for name, traj in all_robot_trajectories[i]:
                if name == traj_name:
                    current_traj = traj
                    break

            if current_traj is None:
                continue

            # Associate and align the trajectories
            try:
                traj_ref, traj_est = sync.associate_trajectories(
                    gt_traj, current_traj, max_diff=0.01)

                # Apply transformation based on mode
                if flag_multi and global_transform is not None:
                    # Apply the global transformation instead of individual alignment
                    traj_est = apply_transformation(traj_est, global_transform)
                else:
                    # For non-multi mode, align each trajectory individually
                    traj_est.align(traj_ref, correct_scale=False,
                                   correct_only_scale=False)

                # Store robot index for color assignment
                color_idx = i % len(COLORS['each'])
                color = COLORS['each'][color_idx]

                # Downsample for plotting
                gt_downsampled = downsample_trajectory(traj_ref, downsample)
                est_downsampled = downsample_trajectory(traj_est, downsample)

                # Plot the ground truth as a black line with transparency
                if i == 0:  # Only add GT label once
                    plot.traj(ax, plot.PlotMode.xy, gt_downsampled,
                              label="真实位置",
                              color='k', alpha=0.3,
                              plot_start_end_markers=False)
                else:
                    plot.traj(ax, plot.PlotMode.xy, gt_downsampled,
                              color='k', alpha=0.3,
                              plot_start_end_markers=False)

                # Plot the estimated trajectory with the robot's color
                plot.traj(ax, plot.PlotMode.xy, est_downsampled,
                          label=f"{robot_name.replace('_', '-')}",
                          color=color,
                          plot_start_end_markers=True)

            except Exception as e:
                print(f"Error aligning trajectory for {robot_name}: {e}")
                continue

        ax.legend()
        ax.grid(True)

        # Save each figure separately
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use a more descriptive filename for each trajectory type, including data_name
        data_suffix = f"_{data_name}" if data_name else ""
        filename = f'trajectory_comparison_{traj_name.replace(" ", "_")}{data_suffix}.pdf'
        fig.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close(fig)
        print(f"Saved comparison plot for {traj_name}")


def plot_global_trajectory(all_gt_trajectories, all_robot_trajectories, robot_names, output_dir, downsample=1):
    """Plot all robot trajectories in a single global view"""
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_title("Global Trajectory View")
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')

    # Collect all ground truth and estimated trajectories
    all_est_trajectories = []

    # First, extract the trajectories in the same order as ground truth
    for i, robot_trajs in enumerate(all_robot_trajectories):
        if not robot_trajs:
            all_est_trajectories.append(None)
            continue

        # Use the first trajectory for each robot in the global view
        all_est_trajectories.append(robot_trajs[0][1])

    # Compute global transformation
    global_transformation = compute_global_transformation(
        all_gt_trajectories, all_est_trajectories)

    # Plot ground truth trajectories with downsampling
    for i, (gt_traj, robot_name) in enumerate(zip(all_gt_trajectories, robot_names)):
        if gt_traj is None:
            continue

        # Downsample for plotting
        gt_downsampled = downsample_trajectory(gt_traj, downsample)

        # Plot ground truth as a black line
        plot.traj(ax, plot.PlotMode.xy, gt_downsampled,
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

        # Downsample for plotting
        est_traj_downsampled = downsample_trajectory(est_traj, downsample)

        # Plot estimated trajectory as solid line
        plot.traj(ax, plot.PlotMode.xy, est_traj_downsampled,
                  label=f"{robot_name}",
                  color=color,
                  plot_start_end_markers=True)

    # Add legend and grid
    ax.legend()
    ax.grid(True)

    # Save figure
    plt.savefig(os.path.join(output_dir, 'global_trajectory.pdf'), dpi=300)
    # plt.savefig(os.path.join(output_dir, 'global_trajectory.png'), dpi=300)
    plt.close(fig)
    print("Saved global trajectory visualization")


def compute_global_transformation(all_gt_trajectories, all_est_trajectories):
    """Compute the global transformation for aligning all trajectories"""
    # Filter out None trajectories
    valid_pairs = [(gt, est) for gt, est in zip(all_gt_trajectories, all_est_trajectories)
                   if gt is not None and est is not None]

    if not valid_pairs:
        return None

    # First associate and align each trajectory pair individually
    aligned_gt_trajs = []
    aligned_est_trajs = []

    for gt_traj, est_traj in valid_pairs:
        try:
            # Associate trajectories
            traj_ref, traj_est = sync.associate_trajectories(
                gt_traj, est_traj, max_diff=0.01)

            # Align the estimated trajectory to its ground truth
            traj_est.align(traj_ref, correct_scale=False,
                           correct_only_scale=False)

            # Store the aligned trajectories
            aligned_gt_trajs.append(traj_ref)
            aligned_est_trajs.append(traj_est)

        except Exception as e:
            print(f"Error aligning a trajectory pair: {e}")
            continue

    if not aligned_gt_trajs or not aligned_est_trajs:
        print("No trajectories could be aligned successfully")
        return None

    # Merge the already aligned trajectories
    merged_gt = merge_trajectories(aligned_gt_trajs)
    merged_est = merge_trajectories(aligned_est_trajs)

    if merged_gt is None or merged_est is None:
        return None

    try:
        # Compute global transformation between merged trajectories
        # Note: No need to associate again since we're already using aligned pairs
        global_transform_r, global_transform_t, _ = merged_est.align(
            merged_gt, correct_scale=False)
        return {
            'rotation': global_transform_r,
            'translation': global_transform_t
        }
    except Exception as e:
        print(f"Error computing global transformation: {e}")
        return None


def create_ape_summary(ape_results, output_dir, combined_metrics=None):
    """Create a summary of APE metrics for all robots and methods"""
    if not ape_results:
        print("No APE results to summarize")
        return

    # Create a DataFrame for the summary
    robot_names = list(ape_results.keys())
    methods = set()
    for robot in ape_results.values():
        methods.update(robot.keys())
    methods = sorted(list(methods))

    # Create data for translation error summary
    trans_data = {'Robot': []}
    for method in methods:
        trans_data[f"{method} (trans)"] = []

    # Create data for full pose error summary
    full_data = {'Robot': []}
    for method in methods:
        full_data[f"{method} (full)"] = []

    # Fill data for each robot
    for robot_name in robot_names:
        trans_data['Robot'].append(robot_name)
        full_data['Robot'].append(robot_name)

        for method in methods:
            if method in ape_results[robot_name]:
                metrics = ape_results[robot_name][method]
                trans_data[f"{method} (trans)"].append(metrics['trans'])
                full_data[f"{method} (full)"].append(metrics['full'])
            else:
                trans_data[f"{method} (trans)"].append(float('nan'))
                full_data[f"{method} (full)"].append(float('nan'))

    # Calculate average across robots
    trans_df = pd.DataFrame(trans_data)
    full_df = pd.DataFrame(full_data)

    trans_df.loc['Average'] = ['Average'] + [trans_df[col].mean()
                                             for col in trans_df.columns[1:]]
    full_df.loc['Average'] = ['Average'] + [full_df[col].mean()
                                            for col in full_df.columns[1:]]

    # If we have combined metrics, add them to the summary
    if combined_metrics:
        # Create a new row dictionary with the correct structure
        combined_trans_row = {'Robot': 'Combined'}
        combined_full_row = {'Robot': 'Combined'}

        for method in methods:
            if method in combined_metrics:
                combined_trans_row[f"{method} (trans)"] = combined_metrics[method]['trans']
                combined_full_row[f"{method} (full)"] = combined_metrics[method]['full']
            else:
                combined_trans_row[f"{method} (trans)"] = float('nan')
                combined_full_row[f"{method} (full)"] = float('nan')

        # Append the rows as new DataFrames with matching structure
        trans_df = pd.concat([trans_df, pd.DataFrame(
            [combined_trans_row])], ignore_index=True)
        full_df = pd.concat([full_df, pd.DataFrame(
            [combined_full_row])], ignore_index=True)

    # Save to CSV
    trans_df.to_csv(os.path.join(
        output_dir, 'ape_translation_summary.csv'), index=False)
    full_df.to_csv(os.path.join(
        output_dir, 'ape_full_summary.csv'), index=False)

    # Create a combined summary with both metrics
    combined_data = {'Robot': trans_data['Robot']}
    for method in methods:
        combined_data[f"{method} (trans)"] = trans_data[f"{method} (trans)"]
        combined_data[f"{method} (full)"] = full_data[f"{method} (full)"]

    combined_df = pd.DataFrame(combined_data)

    # Add combined metrics if available
    if combined_metrics:
        # Create a new row with the same structure as the combined DataFrame
        combined_row = {'Robot': 'Combined'}
        for method in methods:
            if method in combined_metrics:
                combined_row[f"{method} (trans)"] = combined_metrics[method]['trans']
                combined_row[f"{method} (full)"] = combined_metrics[method]['full']
            else:
                combined_row[f"{method} (trans)"] = float('nan')
                combined_row[f"{method} (full)"] = float('nan')

        # Append the row as a new DataFrame
        combined_df = pd.concat(
            [combined_df, pd.DataFrame([combined_row])], ignore_index=True)

    combined_df.to_csv(os.path.join(
        output_dir, 'ape_combined_summary.csv'), index=False)

    print(f"Saved APE summary tables to {output_dir}")

    # Print summary to console
    print("\nAPE Translation Error Summary (RMSE):")
    print(trans_df.to_string(index=False, na_rep='N/A',
          float_format=lambda x: f"{x:.4f}"))

    print("\nAPE Full Pose Error Summary (RMSE):")
    print(full_df.to_string(index=False, na_rep='N/A',
          float_format=lambda x: f"{x:.4f}"))

    # Print combined metrics if available
    if combined_metrics:
        print("\nCombined APE Metrics (RMSE):")
        for method, metrics in combined_metrics.items():
            print(
                f"{method}: Translation={metrics['trans']:.4f}, Full={metrics['full']:.4f}")


def plot_individual_aligned_trajectories(all_gt_trajectories, all_robot_trajectories, robot_names, output_dir, downsample=1, traj_name="", data_name=""):
    """Plot all robot trajectories with each one individually aligned to its own ground truth"""
    # Create figure with 4:3 aspect ratio (12x9 inches maintains the ratio)
    fig, ax = plt.subplots(figsize=(12, 9))
    # ax.set_title("Individually Aligned Trajectories")
    ax.set_xlabel(r'$X$ [m]')
    ax.set_ylabel(r'$Y$ [m]')

    # Instead of collecting points for scatter, collect reference trajectories
    # to plot as lines
    gt_refs = []

    # First pass - align trajectories and collect GT data
    aligned_trajectories = []
    for i, (robot_trajs, robot_name, gt_traj) in enumerate(zip(all_robot_trajectories, robot_names, all_gt_trajectories)):
        if not robot_trajs or gt_traj is None:
            continue

        # Use the first trajectory for each robot
        traj_name, est_traj = robot_trajs[0]

        try:
            # Associate trajectories
            traj_ref, traj_est = sync.associate_trajectories(
                gt_traj, est_traj, max_diff=0.01)

            # Create a new trajectory object using the ASSOCIATED trajectory data
            from evo.core.trajectory import PoseTrajectory3D
            positions = traj_est.positions_xyz.copy()
            orientations = traj_est.orientations_quat_wxyz.copy()
            timestamps = traj_est.timestamps.copy()

            # Create new trajectory object with the associated data
            aligned_est = PoseTrajectory3D(
                positions_xyz=positions,
                orientations_quat_wxyz=orientations,
                timestamps=timestamps
            )

            # Align the estimated trajectory to its own ground truth (also use associated ground truth)
            r, t, s = aligned_est.align(
                traj_ref, correct_scale=False, correct_only_scale=False)

            # Store the reference trajectory for plotting as a line
            gt_refs.append(traj_ref)

            # Store aligned trajectory for plotting
            color = COLORS['each'][i % len(COLORS['each'])]
            aligned_trajectories.append((robot_name, aligned_est, color))

        except Exception as e:
            print(f"Error aligning trajectory for {robot_name}: {e}")
            continue

    # Plot the ground truth trajectories as lines with a single label
    if gt_refs:
        # Downsample the ground truth trajectories for plotting
        gt_refs_downsampled = [downsample_trajectory(
            gt_ref, downsample) for gt_ref in gt_refs]

        # Plot first GT trajectory with label
        plot.traj(ax, plot.PlotMode.xy, gt_refs_downsampled[0],
                  label="真实位置",
                  color='k', alpha=0.5,
                  plot_start_end_markers=False)

        # Plot remaining GT trajectories without labels
        for gt_ref in gt_refs_downsampled[1:]:
            plot.traj(ax, plot.PlotMode.xy, gt_ref,
                      color='k', alpha=0.5,
                      plot_start_end_markers=False)

    # Plot all aligned trajectories with downsampling
    for robot_name, aligned_est, color in aligned_trajectories:
        # Downsample the estimated trajectory
        downsampled_est = downsample_trajectory(aligned_est, downsample)

        plot.traj(ax, plot.PlotMode.xy, downsampled_est,
                  label=f"{robot_name.replace('_','-')}",
                  color=color,
                  plot_start_end_markers=True)

    # Add legend and grid
    # 'lower right' corresponds to southeast position
    ax.legend(loc='upper left')
    # ax.legend()
    ax.grid(True)

    # Save figure with trajectory name in the filename
    filename_base = 'individual_aligned_trajectories'
    if traj_name:
        filename_base = f'individual_aligned_{data_name}_{traj_name.replace(" ", "_")}'

    plt.savefig(os.path.join(
        output_dir, f'{filename_base}.pdf'), dpi=300)
    # plt.savefig(os.path.join(
    #     output_dir, f'{filename_base}.png'), dpi=300)
    plt.close(fig)
    print(
        f"Saved individually aligned trajectories visualization for {traj_name if traj_name else 'all trajectories'}")


def downsample_trajectory(trajectory, factor):
    """Downsample a trajectory by the given factor to reduce plotting complexity"""
    if factor <= 1 or trajectory is None:
        return trajectory

    from evo.core.trajectory import PoseTrajectory3D

    # Select every Nth point
    indices = range(0, len(trajectory.positions_xyz), factor)

    if len(indices) < 2:  # Ensure we have at least 2 points
        indices = [0, len(trajectory.positions_xyz)-1]

    # Create a new downsampled trajectory
    positions = trajectory.positions_xyz[indices].copy()
    orientations = trajectory.orientations_quat_wxyz[indices].copy()
    timestamps = trajectory.timestamps[indices].copy()

    downsampled = PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=orientations,
        timestamps=timestamps
    )

    return downsampled


def calculate_ape_metrics(gt_traj, est_traj, global_transform=None):
    """Calculate APE metrics for a given ground truth and estimated trajectory pair"""
    if gt_traj is None or est_traj is None:
        return None

    try:
        # Associate trajectories for evaluation
        traj_ref, traj_est = sync.associate_trajectories(
            gt_traj, est_traj, max_diff=0.01)

        # If global_transform is provided, apply it first (for multi-robot mode)
        if global_transform is not None:
            # Create a proper copy of the trajectory instead of using .copy()
            from evo.core.trajectory import PoseTrajectory3D
            positions = traj_est.positions_xyz.copy()
            orientations = traj_est.orientations_quat_wxyz.copy()
            timestamps = traj_est.timestamps.copy()

            traj_est_copy = PoseTrajectory3D(
                positions_xyz=positions,
                orientations_quat_wxyz=orientations,
                timestamps=timestamps
            )

            traj_est = apply_transformation(traj_est_copy, global_transform)
        else:
            # Otherwise use individual alignment (for single robot mode)
            traj_est.align(traj_ref, correct_scale=False,
                           correct_only_scale=False)

        # Calculate APE metrics
        ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
        ape_full = metrics.APE(metrics.PoseRelation.full_transformation)

        ape_trans.process_data((traj_ref, traj_est))
        ape_full.process_data((traj_ref, traj_est))

        return {
            'trans': ape_trans.get_statistic(metrics.StatisticsType.rmse),
            'full': ape_full.get_statistic(metrics.StatisticsType.rmse),
            'path_length': traj_ref.path_length
        }
    except Exception as e:
        print(f"Error calculating APE metrics: {e}")
        return None


def calculate_combined_ape_metrics(all_gt_trajectories, all_robot_trajectories, method_names, flag_multi=False):
    """
    Calculate APE metrics for combined trajectories.
    For each method, align and merge all robot trajectories, then calculate APE against merged GT.
    If flag_multi is True, merge first and then align (for collaborative/distributed SLAM)
    If flag_multi is False, align each trajectory pair first and then merge (for single robot SLAM)
    """
    combined_metrics = {}

    # For each method, extract and merge all robot trajectories
    all_method_names = set()
    for robot_trajectories in all_robot_trajectories:
        for name, _ in robot_trajectories:
            all_method_names.add(name)

    for method_name in all_method_names:
        if flag_multi:
            # For multi-robot mode: First associate trajectories, then merge all, then compute single alignment
            all_associated_gt_trajs = []
            all_associated_est_trajs = []

            # Process each robot
            for i, robot_trajectories in enumerate(all_robot_trajectories):
                if all_gt_trajectories[i] is None:
                    continue

                # Find the trajectory for this method
                gt_traj = all_gt_trajectories[i]
                est_traj = None
                for name, traj in robot_trajectories:
                    if name == method_name:
                        est_traj = traj
                        break

                if est_traj is None:
                    continue

                # Associate the trajectories (but don't align yet)
                try:
                    traj_ref, traj_est = sync.associate_trajectories(
                        gt_traj, est_traj, max_diff=0.01)

                    # Store the associated trajectories
                    all_associated_gt_trajs.append(traj_ref)
                    all_associated_est_trajs.append(traj_est)

                except Exception as e:
                    print(
                        f"Error associating trajectory for {method_name}: {e}")
                    continue

            if not all_associated_gt_trajs or not all_associated_est_trajs:
                print(f"No valid trajectories found for method {method_name}")
                continue

            # Merge all associated trajectories for this method
            merged_gt = merge_trajectories(all_associated_gt_trajs)
            merged_est = merge_trajectories(all_associated_est_trajs)

            if merged_gt is None or merged_est is None:
                print(f"Failed to merge trajectories for method {method_name}")
                continue

            # Now compute a single global alignment transformation
            try:
                r, t, s = merged_est.align(merged_gt, correct_scale=False,
                                           correct_only_scale=False)

                global_transform = {
                    'rotation': r,
                    'translation': t
                }

                # Apply global transformation to all individual trajectories
                transformed_est_trajs = []
                for i, est_traj in enumerate(all_associated_est_trajs):
                    # Create proper copy for transformation
                    from evo.core.trajectory import PoseTrajectory3D
                    positions = est_traj.positions_xyz.copy()
                    orientations = est_traj.orientations_quat_wxyz.copy()
                    timestamps = est_traj.timestamps.copy()

                    est_traj_copy = PoseTrajectory3D(
                        positions_xyz=positions,
                        orientations_quat_wxyz=orientations,
                        timestamps=timestamps
                    )

                    # Apply the global transformation
                    transformed_traj = apply_transformation(
                        est_traj_copy, global_transform)
                    transformed_est_trajs.append(transformed_traj)

                # Calculate APE metrics using transformed trajectories
                all_errors_trans = []
                all_errors_full = []

                for gt_traj, est_traj in zip(all_associated_gt_trajs, transformed_est_trajs):
                    # Calculate individual APE without further alignment
                    ape_trans = metrics.APE(
                        metrics.PoseRelation.translation_part)
                    ape_full = metrics.APE(
                        metrics.PoseRelation.full_transformation)

                    ape_trans.process_data((gt_traj, est_traj))
                    ape_full.process_data((gt_traj, est_traj))

                    all_errors_trans.append(
                        ape_trans.get_statistic(metrics.StatisticsType.rmse))
                    all_errors_full.append(
                        ape_full.get_statistic(metrics.StatisticsType.rmse))

                # Calculate combined metrics
                # Use merged trajectory for the final metrics
                merged_transformed_est = merge_trajectories(
                    transformed_est_trajs)

                ape_trans_final = metrics.APE(
                    metrics.PoseRelation.translation_part)
                ape_full_final = metrics.APE(
                    metrics.PoseRelation.full_transformation)

                ape_trans_final.process_data(
                    (merged_gt, merged_transformed_est))
                ape_full_final.process_data(
                    (merged_gt, merged_transformed_est))

                combined_metrics[method_name] = {
                    'trans': ape_trans_final.get_statistic(metrics.StatisticsType.rmse),
                    'full': ape_full_final.get_statistic(metrics.StatisticsType.rmse),
                    'path_length': merged_gt.path_length
                }

                print(f"Combined APE for {method_name} (global transformation): Trans={combined_metrics[method_name]['trans']:.4f}, "
                      f"Full={combined_metrics[method_name]['full']:.4f}")

            except Exception as e:
                print(
                    f"Error calculating global metrics for method {method_name}: {e}")
        else:
            # For single robot mode - align each trajectory pair first, then merge
            # For each method, collect and align trajectory pairs
            aligned_gt_trajs = []
            aligned_est_trajs = []

            # Process each robot
            for i, robot_trajectories in enumerate(all_robot_trajectories):
                if all_gt_trajectories[i] is None:
                    continue

                # Find the trajectory for this method
                gt_traj = all_gt_trajectories[i]
                est_traj = None
                for name, traj in robot_trajectories:
                    if name == method_name:
                        est_traj = traj
                        break

                if est_traj is None:
                    continue

                # Associate and align the trajectories
                try:
                    traj_ref, traj_est = sync.associate_trajectories(
                        gt_traj, est_traj, max_diff=0.01)

                    # Align the estimated trajectory to ground truth
                    traj_est.align(traj_ref, correct_scale=False,
                                   correct_only_scale=False)

                    # Store the aligned trajectories
                    aligned_gt_trajs.append(traj_ref)
                    aligned_est_trajs.append(traj_est)

                except Exception as e:
                    print(f"Error aligning trajectory for {method_name}: {e}")
                    continue

            if not aligned_gt_trajs or not aligned_est_trajs:
                print(
                    f"No valid aligned trajectories found for method {method_name}")
                continue

            # Merge the already aligned trajectories
            merged_gt = merge_trajectories(aligned_gt_trajs)
            merged_est = merge_trajectories(aligned_est_trajs)

            if merged_gt is None or merged_est is None:
                print(f"Failed to merge trajectories for method {method_name}")
                continue

            # Calculate APE metrics on the merged trajectories directly
            # No need to associate or align again as we're already using aligned pairs
            try:
                # Calculate APE metrics
                ape_trans = metrics.APE(metrics.PoseRelation.translation_part)
                ape_full = metrics.APE(
                    metrics.PoseRelation.full_transformation)

                ape_trans.process_data((merged_gt, merged_est))
                ape_full.process_data((merged_gt, merged_est))

                combined_metrics[method_name] = {
                    'trans': ape_trans.get_statistic(metrics.StatisticsType.rmse),
                    'full': ape_full.get_statistic(metrics.StatisticsType.rmse),
                    'path_length': merged_gt.path_length
                }

                print(f"Combined APE for {method_name} (align-then-merge): Trans={combined_metrics[method_name]['trans']:.4f}, "
                      f"Full={combined_metrics[method_name]['full']:.4f}")

            except Exception as e:
                print(
                    f"Error calculating combined APE metrics for method {method_name}: {e}")

    return combined_metrics


def main():
    args = parse_args()

    gt_dir = args.gt_dir
    output_dir = args.output_dir
    flag_multi = args.flag_multi

    # Filter out empty trajectory directories
    traj_dirs = [d for d in [args.traj_dir1, args.traj_dir2,
                             args.traj_dir3, args.traj_dir4] if d]
    traj_names = [args.traj_name1, args.traj_name2,
                  args.traj_name3, args.traj_name4][:len(traj_dirs)]

    # Extract robot names from directories in traj_dir1 instead of using predefined names
    robot_names = []
    if os.path.exists(args.traj_dir1):
        # List all directories in traj_dir1
        subdirs = [d for d in os.listdir(args.traj_dir1) if os.path.isdir(
            os.path.join(args.traj_dir1, d))]

        # Filter out common non-robot directories
        robot_names = [d for d in subdirs if not d.startswith('.') and d not in [
            'log', 'config']]

        # Limit to specified number if provided
        if args.robot_num < len(robot_names):
            robot_names = robot_names[:args.robot_num]

    # Fallback to predefined names if no robots found
    if not robot_names:
        print("No robot directories found in traj_dir1, using predefined robot names")
        robot_names = ROBOT_NAMES[:args.robot_num]
    else:
        print(f"Found robot directories: {', '.join(robot_names)}")

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read ground truth trajectories
    gt_trajectories = read_ground_truth(gt_dir, robot_names, args.dataset)

    # Store all trajectories for combined plot
    all_robot_trajectories = []

    # Dictionary to store APE results for each robot and method
    ape_results = {}

    # First pass to collect all trajectories
    for i, robot_name in enumerate(robot_names):
        # Read trajectories for this robot from multiple directories
        robot_trajectories = read_robot_trajectories(
            robot_name, traj_dirs, traj_names, args.traj_file_pattern, flag_multi)
        all_robot_trajectories.append(robot_trajectories)

    # For multi-robot mode, compute global transformations for each method
    global_transforms = {}
    if flag_multi:
        print("Multi-robot mode: Computing global transformations for each method...")
        # Extract all unique method names
        all_method_names = set()
        for robot_trajectories in all_robot_trajectories:
            for name, _ in robot_trajectories:
                all_method_names.add(name)

        # For each method, compute a global transformation
        for method_name in all_method_names:
            # Collect all trajectories for this method
            method_gt_trajectories = []
            method_est_trajectories = []

            # First associate each trajectory pair
            associated_gt_trajs = []
            associated_est_trajs = []

            for i, (robot_trajectories, gt_traj) in enumerate(zip(all_robot_trajectories, gt_trajectories)):
                if gt_traj is None:
                    continue

                # Find this method's trajectory for current robot
                est_traj = None
                for name, traj in robot_trajectories:
                    if name == method_name:
                        est_traj = traj
                        break

                if est_traj is None:
                    continue

                # Associate trajectories (but don't align yet)
                try:
                    traj_ref, traj_est = sync.associate_trajectories(
                        gt_traj, est_traj, max_diff=0.01)

                    # Store the associated trajectories
                    associated_gt_trajs.append(traj_ref)
                    associated_est_trajs.append(traj_est)

                except Exception as e:
                    print(
                        f"Error associating trajectory for robot {i} and method {method_name}: {e}")
                    continue

            # Merge all associated trajectories
            if associated_gt_trajs and associated_est_trajs:
                merged_gt = merge_trajectories(associated_gt_trajs)
                merged_est = merge_trajectories(associated_est_trajs)

                if merged_gt is not None and merged_est is not None:
                    # Compute a single global transformation from the merged trajectories
                    global_transforms[method_name] = compute_transformation(
                        merged_gt, merged_est)

                    if global_transforms[method_name] is not None:
                        print(
                            f"Computed global transformation for method '{method_name}'")
                    else:
                        print(
                            f"Failed to compute global transformation for method '{method_name}'")
                else:
                    print(
                        f"Failed to merge trajectories for method '{method_name}'")
            else:
                print(
                    f"No valid associated trajectory pairs for method '{method_name}'")

    # Process each robot for APE calculations and plotting
    for i, robot_name in enumerate(robot_names):
        print(f"Processing robot: {robot_name}")

        # Retrieve previously collected trajectories
        robot_trajectories = all_robot_trajectories[i]

        # Calculate APE metrics for each trajectory
        if gt_trajectories[i] is not None:
            ape_results[robot_name] = {}
            for traj_name, traj in robot_trajectories:
                # For multi-robot mode, use global transformation
                if flag_multi and traj_name in global_transforms:
                    metrics = calculate_ape_metrics(
                        gt_trajectories[i], traj, global_transforms[traj_name])
                else:
                    metrics = calculate_ape_metrics(gt_trajectories[i], traj)

                if metrics:
                    ape_results[robot_name][traj_name] = metrics
                    print(
                        f"{robot_name} - {traj_name}: APE Trans={metrics['trans']:.4f}, APE Full={metrics['full']:.4f}")

        # Plot individual robot trajectories
        plot_robot_trajectories(
            gt_trajectories[i], robot_trajectories, robot_name, output_dir, args.downsample)

    # Create global trajectory plot with all robots using appropriate method based on flag_multi
    if not flag_multi:
        # plot_global_trajectory(gt_trajectories, all_robot_trajectories,
        #                        robot_names, output_dir, args.downsample)
        plot_individual_aligned_trajectories(gt_trajectories, all_robot_trajectories,
                                             robot_names, output_dir, args.downsample,
                                             traj_name=args.traj_name1, data_name=args.gt_dir[-5:-1])
    # else:
    #     # For single robot mode, plot both global alignment and individual alignments
    #     plot_global_trajectory(gt_trajectories, all_robot_trajectories,
    #                            robot_names, output_dir, args.downsample)

    # Create APE summary tables
    combined_metrics = None
    # Calculate combined APE metrics for both multi and non-multi modes
    combined_metrics = calculate_combined_ape_metrics(
        gt_trajectories, all_robot_trajectories, traj_names, flag_multi)

    create_ape_summary(ape_results, output_dir, combined_metrics)

    # Only create comparison plots if we have trajectories from at least one method
    if traj_dirs:
        # Use the last trajectory as reference if available, otherwise use the first one
        ref_traj_name = traj_names[-1] if len(traj_names) > 0 else None

        # Extract data name from ground truth directory for filename
        data_name = args.gt_dir[-5:-1]  # Extract "1207" or "1208" from path

        # Create plots with all robots for each trajectory type
        plot_all_robot_trajectories(gt_trajectories, all_robot_trajectories,
                                    robot_names, output_dir,
                                    ref_robot_name='acl_jackal',
                                    ref_traj_name=ref_traj_name,
                                    downsample=args.downsample,
                                    data_name=data_name,
                                    flag_multi=flag_multi)  # Pass data_name to function

    print(f"All trajectory plots and metrics saved to {output_dir}")


if __name__ == '__main__':
    main()
