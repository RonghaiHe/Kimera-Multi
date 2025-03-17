'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-03-01
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-03-17 19:00:10
FilePath: /src/kimera_multi/examples/plot_multi_trajectory.py
Description:
  Script to read multiple trajectory files and plot them for comparison.
  For each robot, plot its ground truth and multiple estimated trajectories.
'''
import scienceplots
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
# import glob
import argparse
import numpy as np
from evo.tools import file_interface
from evo.core import sync
from evo.tools import plot
from evo.tools.settings import SETTINGS


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
               'sparkal1', 'sparkal2', 'hathor', 'thoth']

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
    parser.add_argument('--robot_num', type=int, default=6,
                        help='Number of robots to process')
    parser.add_argument('--dataset', type=str, default='Kimera-Multi',
                        help='Name of the dataset')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground truth directory path')
    parser.add_argument('--output_dir', type=str, required=False, default='./traj',
                        help='Output directory for trajectory plots')

    # Make trajectory directories optional with a minimum of 1
    parser.add_argument('--traj_dir1', type=str, required=True,
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
    parser.add_argument('--traj_file_pattern', type=str, default='distributed/kimera_distributed_poses_tum_2500.tum',
                        help='Pattern for trajectory file relative to traj_dir/robot_name/')

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


def read_robot_trajectories(robot_name, traj_dirs, traj_names, file_pattern):
    """Read trajectories for a single robot from multiple directories"""
    trajectories = []

    for i, traj_dir in enumerate(traj_dirs):
        if not traj_dir:  # Skip empty directory arguments
            continue

        # Allow for custom file structure
        traj_path = os.path.join(traj_dir, robot_name, file_pattern)
        traj = read_robot_trajectory(traj_path)
        if traj is not None:
            trajectories.append((traj_names[i], traj))
            print(
                f"Loaded trajectory {traj_names[i]} for {robot_name} from {traj_path}")

    return trajectories


def plot_robot_trajectories(gt_traj, robot_trajectories, robot_name, output_dir):
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

    # Plot ground truth
    plot.traj(ax, plot.PlotMode.xy, gt_traj, label='Ground Truth',
              color=COLORS['gt'], plot_start_end_markers=True)

    # Plot estimated trajectories
    for i, (traj_name, traj_est) in enumerate(robot_trajectories):
        try:
            traj_ref, traj_est = sync.associate_trajectories(
                gt_traj, traj_est, max_diff=0.01)
            traj_est.align(traj_ref, correct_scale=False,
                           correct_only_scale=False)

            # Choose color from the each robots' colors list
            color_idx = i % len(COLORS['each'])
            color = COLORS['each'][color_idx]

            plot.traj(ax, plot.PlotMode.xy, traj_est, label=traj_name.replace('_', '-'),
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
        output_dir, f'{robot_name}_trajectories.pdf'), dpi=300)  # , bbox_inches='tight')
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


def plot_all_robot_trajectories(all_gt_trajectories, all_robot_trajectories, robot_names, output_dir, ref_robot_name=None, ref_traj_name=None):
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

        # Collect all ground truth and estimated trajectories for this method
        all_gt_for_method = []
        all_est_for_method = []

        for i, (robot_name, gt_traj) in enumerate(zip(robot_names, all_gt_trajectories)):
            if gt_traj is None:
                continue

            # Find this trajectory type for the current robot
            current_traj = None
            for name, traj in all_robot_trajectories[i]:
                if name == traj_name:
                    current_traj = traj
                    break

            if current_traj is not None:
                all_gt_for_method.append(gt_traj)
                all_est_for_method.append(current_traj)

        if not all_gt_for_method or not all_est_for_method:
            print(f"No valid trajectories found for method {traj_name}")
            continue

        try:
            # Merge all ground truth and estimated trajectories
            merged_gt = merge_trajectories(all_gt_for_method)
            merged_est = merge_trajectories(all_est_for_method)

            if merged_gt is None or merged_est is None:
                print(f"Failed to merge trajectories for {traj_name}")
                continue

            # Align the merged estimated trajectory to the merged ground truth
            # This computes the global transformation for all robots together
            aligned_gt, aligned_est = sync.associate_trajectories(
                merged_gt, merged_est, max_diff=0.01)
            global_transform_r, global_transform_t, _ = aligned_est.align(
                aligned_gt, correct_scale=False)
            global_transformation = {
                'rotation': global_transform_r,
                'translation': global_transform_t
            }

            # Plot merged ground truth trajectories as dots
            plt.scatter(merged_gt.positions_xyz[:, 0], merged_gt.positions_xyz[:, 1],
                        s=1, c='black', alpha=0.3, label="真实位置")

            # Plot each robot's trajectory with the global transformation
            for i, (robot_name, gt_traj) in enumerate(zip(robot_names, all_gt_trajectories)):
                if gt_traj is None:
                    continue

                robot_trajs = all_robot_trajectories[i]
                current_traj = None
                for name, traj in robot_trajs:
                    if name == traj_name:
                        current_traj = traj
                        break

                if current_traj is None:
                    continue

                # Apply the global transformation to each robot's trajectory
                traj_est = apply_transformation(
                    current_traj, global_transformation)
                color_idx = i % len(COLORS['each'])
                color = COLORS['each'][color_idx]

                plot.traj(ax, plot.PlotMode.xy, traj_est, label=f"{robot_name.replace('_', '-')}",
                          color=color, plot_start_end_markers=True)

        except Exception as e:
            print(f"Error processing method {traj_name}: {e}")
            continue

        ax.legend()
        ax.grid(True)

        # Save each figure separately
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use a more descriptive filename for each trajectory type
        filename = f'trajectory_comparison_between_factor_{traj_name.replace(" ", "_")}.pdf'
        fig.savefig(os.path.join(output_dir, filename),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved comparison plot for {traj_name}")


def main():
    args = parse_args()

    gt_dir = args.gt_dir
    output_dir = args.output_dir

    # Filter out empty trajectory directories
    traj_dirs = [d for d in [args.traj_dir1, args.traj_dir2,
                             args.traj_dir3, args.traj_dir4] if d]
    traj_names = [args.traj_name1, args.traj_name2,
                  args.traj_name3, args.traj_name4][:len(traj_dirs)]

    # Limit robots based on input
    robot_names = ROBOT_NAMES[:args.robot_num]

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read ground truth trajectories
    gt_trajectories = read_ground_truth(gt_dir, robot_names, args.dataset)

    # Store all trajectories for combined plot
    all_robot_trajectories = []

    # Process each robot
    for i, robot_name in enumerate(robot_names):
        print(f"Processing robot: {robot_name}")

        # Read trajectories for this robot from multiple directories
        robot_trajectories = read_robot_trajectories(
            robot_name, traj_dirs, traj_names, args.traj_file_pattern)

        # Store for later combined plot
        all_robot_trajectories.append(robot_trajectories)

        # Plot individual robot trajectories
        # plot_robot_trajectories(
        #     gt_trajectories[i], robot_trajectories, robot_name, output_dir)

    # Only create comparison plots if we have trajectories from at least one method
    if traj_dirs:
        # Use the last trajectory as reference if available, otherwise use the first one
        ref_traj_name = traj_names[-1] if len(traj_names) > 0 else None

        # Create plots with all robots for each trajectory type
        plot_all_robot_trajectories(gt_trajectories, all_robot_trajectories,
                                    robot_names, output_dir,
                                    ref_robot_name='acl_jackal',
                                    ref_traj_name=ref_traj_name)

    print(f"All trajectory plots saved to {output_dir}")


if __name__ == '__main__':
    main()
