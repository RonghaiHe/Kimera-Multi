'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-03-20
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-04-17 21:55:49
FilePath: /src/kimera_multi/examples/plot_time_filtered_trajectory.py
Description:
  Script to read ground truth trajectory files for selected robots and plot them with time filtering.
  Allows visualization of trajectories within specific time windows.
'''
import scienceplots
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import datetime
from evo.tools import file_interface
from evo.core import sync
from evo.tools import plot
from evo.tools.settings import SETTINGS
from evo.core.trajectory import PoseTrajectory3D

'''
Usage examples:
1. Using seconds:
python plot_time_filtered_trajectory.py --gt_dir /path/to/gt \
    --robots acl_jackal sparkal1 \
    --start_time 10.5 --end_time 30.0 \
    --plot_3d --title "Robot Navigation Segment"

2. Using timestamps:
python plot_time_filtered_trajectory.py --gt_dir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/1208/ \
    --start_timestamp "1670533260" --end_timestamp "1670534520" \
    --plot_3d --title "Robot Navigation Segment"
'''

SETTINGS.plot_usetex = False
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
               'sparkal1', 'sparkal2', 'hathor', 'thoth', 'apis', 'sobek']

# Colors for different robots
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', 'k']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot trajectories for selected robots with time filtering')
    parser.add_argument('--gt_dir', type=str, required=True,
                        help='Ground truth directory path')
    parser.add_argument('--output_dir', type=str, required=False, default='./traj_filtered',
                        help='Output directory for trajectory plots')
    parser.add_argument('--dataset', type=str, default='Kimera-Multi',
                        help='Name of the dataset (Kimera-Multi or EuRoC)')
    parser.add_argument('--robots', type=str, nargs='+', default=ROBOT_NAMES,
                        help='List of robots to include in the visualization')

    # Time filtering options - support both seconds and timestamps
    time_group = parser.add_argument_group('Time filtering options')
    time_group.add_argument('--start_time', type=float, default=None,
                            help='Start time for trajectory filtering in seconds (relative to trajectory start)')
    time_group.add_argument('--end_time', type=float, default=None,
                            help='End time for trajectory filtering in seconds (relative to trajectory start)')
    time_group.add_argument('--start_timestamp', type=str, default=None,
                            help='Start time as absolute timestamp (e.g., "1403636579.195703096")')
    time_group.add_argument('--end_timestamp', type=str, default=None,
                            help='End time as absolute timestamp (e.g., "1403636589.987654321")')

    # Visualization options
    parser.add_argument('--individual', action='store_true',
                        help='Generate individual plots for each robot (default: combined plot)')
    parser.add_argument('--plot_3d', action='store_true',
                        help='Generate 3D plots instead of 2D')
    parser.add_argument('--title', type=str, default='Robot Trajectories',
                        help='Plot title')

    args = parser.parse_args()

    # Convert string timestamps to float if provided
    if args.start_timestamp:
        try:
            args.start_time = float(args.start_timestamp)
            print(f"Using absolute start timestamp: {args.start_time}")
        except ValueError:
            print(
                f"Error: Could not convert start_timestamp '{args.start_timestamp}' to float")

    if args.end_timestamp:
        try:
            args.end_time = float(args.end_timestamp)
            print(f"Using absolute end timestamp: {args.end_time}")
        except ValueError:
            print(
                f"Error: Could not convert end_timestamp '{args.end_timestamp}' to float")

    return args


def read_ground_truth(gt_dir, robot_names, dataset_name):
    """Read ground truth trajectories for all requested robots"""
    print("Loading ground truth trajectories...")
    gt_trajectories = {}

    for robot_name in robot_names:
        if dataset_name == 'Kimera-Multi':
            ref_file = os.path.join(
                gt_dir, f'modified_{robot_name}_gt_odom.tum')
            # Check for CSV file if TUM file doesn't exist
            if not os.path.exists(ref_file):
                ref_file = os.path.join(
                    gt_dir, f'modified_{robot_name}_gt_odom.csv')
        elif dataset_name == 'EuRoC':
            ref_file = os.path.join(gt_dir, 'data.tum')
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        if os.path.exists(ref_file):
            try:
                if ref_file.endswith('.csv'):
                    traj = read_csv_trajectory_file(ref_file)
                else:
                    traj = file_interface.read_tum_trajectory_file(ref_file)
                gt_trajectories[robot_name] = traj
                print(f"Loaded GT trajectory for {robot_name}")
            except Exception as e:
                print(f"Error loading GT trajectory for {robot_name}: {e}")
                gt_trajectories[robot_name] = None
        else:
            print(f"GT trajectory file not found for {robot_name}: {ref_file}")
            gt_trajectories[robot_name] = None

    return gt_trajectories


def read_csv_trajectory_file(file_path):
    """Read trajectory from CSV file with header"""
    import numpy as np
    import pandas as pd
    from evo.core.trajectory import PoseTrajectory3D

    print(f"Reading CSV trajectory from {file_path}")

    # Read CSV file with pandas to handle headers properly
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Read the file with pandas, use '#' as comment character
        df = pd.read_csv(file_path, comment='#')

        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")

        # Print first few rows to debug
        print(f"First few rows of CSV data:")
        print(df.head())

        # Check column headers and extract proper columns
        headers = list(df.columns)
        print(f"Found headers: {headers}")

        # Determine which columns to use based on headers
        # Check if we need to map common column name patterns
        timestamp_patterns = ['timestamp', 'time', 'timestamp_kf']
        timestamps_col = None

        # Try to find timestamp column
        for pattern in timestamp_patterns:
            matching_cols = [col for col in headers if pattern in col.lower()]
            if matching_cols:
                timestamps_col = matching_cols[0]
                break

        # If still not found, use first column
        if timestamps_col is None:
            timestamps_col = headers[0]
            print(
                f"  No explicit timestamp column found, using first column: {timestamps_col}")
        else:
            print(f"  Using timestamp column: {timestamps_col}")

        # Position columns (x, y, z)
        x_col = headers[1] if len(headers) > 1 else None
        y_col = headers[2] if len(headers) > 2 else None
        z_col = headers[3] if len(headers) > 3 else None

        # Quaternion columns (qx, qy, qz, qw)
        qx_col = headers[4] if len(headers) > 4 else None
        qy_col = headers[5] if len(headers) > 5 else None
        qz_col = headers[6] if len(headers) > 6 else None
        qw_col = headers[7] if len(headers) > 7 else None

        print(f"  Using columns: timestamp={timestamps_col}, x={x_col}, y={y_col}, z={z_col}, "
              f"qx={qx_col}, qy={qy_col}, qz={qz_col}, qw={qw_col}")

        # Check that all required columns exist
        required_cols = [timestamps_col, x_col, y_col,
                         z_col, qx_col, qy_col, qz_col, qw_col]
        if None in required_cols:
            missing = [name for i, name in enumerate(['timestamp', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
                       if required_cols[i] is None]
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        # Convert data to arrays needed by PoseTrajectory3D
        # Use more robust conversion with error handling
        try:
            timestamps = df[timestamps_col].values.astype(float)
            print(
                f"  Timestamps range: {min(timestamps):.2f} to {max(timestamps):.2f}")
        except Exception as e:
            print(f"  Error converting timestamps: {e}")
            print(f"  Sample timestamp values: {df[timestamps_col].head()}")
            raise ValueError(
                f"Invalid timestamp data in column {timestamps_col}")

        # Handle empty DataFrame
        if len(timestamps) == 0:
            raise ValueError(
                f"No timestamp data found in CSV file: {file_path}")

        # Extract and convert position data
        try:
            positions_xyz = np.column_stack([
                df[x_col].values.astype(float),
                df[y_col].values.astype(float),
                df[z_col].values.astype(float)
            ])
        except Exception as e:
            print(f"  Error converting position data: {e}")
            print(
                f"  Sample position values: x={df[x_col].head()}, y={df[y_col].head()}, z={df[z_col].head()}")
            raise ValueError("Invalid position data")

        # Extract and convert quaternion data
        try:
            # Note: evo expects quaternions in w, x, y, z order while most CSV files store them as x, y, z, w
            orientations_quat_wxyz = np.column_stack([
                df[qw_col].values.astype(float),  # w component first
                df[qx_col].values.astype(float),
                df[qy_col].values.astype(float),
                df[qz_col].values.astype(float)
            ])
        except Exception as e:
            print(f"  Error converting quaternion data: {e}")
            print(
                f"  Sample quaternion values: qw={df[qw_col].head()}, qx={df[qx_col].head()}, qy={df[qy_col].head()}, qz={df[qz_col].head()}")
            raise ValueError("Invalid quaternion data")

        print(
            f"  Data shapes: timestamps={timestamps.shape}, positions={positions_xyz.shape}, orientations={orientations_quat_wxyz.shape}")

        # Create PoseTrajectory3D object
        traj = PoseTrajectory3D(
            positions_xyz=positions_xyz,
            orientations_quat_wxyz=orientations_quat_wxyz,
            timestamps=timestamps
        )

        print(f"  Successfully loaded {len(timestamps)} poses from CSV")
        return traj

    except Exception as e:
        print(f"Error parsing CSV file: {e}")
        import traceback
        traceback.print_exc()

        # Return a dummy trajectory with minimal data to avoid crashing
        print("Creating a dummy trajectory with minimal data")
        dummy_timestamp = np.array([0.0])
        dummy_position = np.array([[0.0, 0.0, 0.0]])
        # Identity quaternion (w,x,y,z)
        dummy_orientation = np.array([[1.0, 0.0, 0.0, 0.0]])

        return PoseTrajectory3D(
            positions_xyz=dummy_position,
            orientations_quat_wxyz=dummy_orientation,
            timestamps=dummy_timestamp
        )


def filter_trajectory_by_time(trajectory, start_time=None, end_time=None):
    """Filter a trajectory to include only poses within the specified time range"""
    if trajectory is None:
        return None

    if start_time is None and end_time is None:
        return trajectory

    # Print timestamp range information for debugging
    if len(trajectory.timestamps) > 0:
        print(
            f"  Original trajectory has {len(trajectory.timestamps)} timestamps")
        print(
            f"  Timestamp range: {trajectory.timestamps[0]:.2f} to {trajectory.timestamps[-1]:.2f}")
    else:
        print(f"  Warning: Trajectory has 0 timestamps!")
        return trajectory  # Return original trajectory to avoid empty data error

    # Initialize filter indices
    mask = np.ones(len(trajectory.timestamps), dtype=bool)

    # Apply start time filter if provided
    if start_time is not None:
        mask = np.logical_and(mask, trajectory.timestamps >= start_time)
        print(f"  After start_time filter: {np.sum(mask)} timestamps left")

    # Apply end time filter if provided
    if end_time is not None:
        mask = np.logical_and(mask, trajectory.timestamps <= end_time)
        print(f"  After end_time filter: {np.sum(mask)} timestamps left")

    # Check if we have any data left
    if np.sum(mask) == 0:
        print(f"  Warning: No data left after filtering! Check your time range.")
        # Create a minimal valid trajectory with at least one point to avoid empty data error
        filtered_timestamps = np.array([trajectory.timestamps[0]])
        filtered_positions = trajectory.positions_xyz[0:1]
        filtered_orientations = trajectory.orientations_quat_wxyz[0:1]
    else:
        # Create filtered trajectory
        filtered_timestamps = trajectory.timestamps[mask]
        filtered_positions = trajectory.positions_xyz[mask]
        filtered_orientations = trajectory.orientations_quat_wxyz[mask]

    # Return a new trajectory with the filtered data
    filtered_traj = PoseTrajectory3D(
        positions_xyz=filtered_positions,
        orientations_quat_wxyz=filtered_orientations,
        timestamps=filtered_timestamps
    )

    return filtered_traj


def plot_trajectories_2d(trajectories, robot_names, output_dir, title, combined=True):
    """Plot 2D trajectories, either combined or individually"""

    if combined:
        # Plot all trajectories on one figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title)
        ax.set_xlabel(r'$X$ [m]')
        ax.set_ylabel(r'$Y$ [m]')

        for i, robot_name in enumerate(robot_names):
            traj = trajectories.get(robot_name)
            if traj is None:
                continue

            color = COLORS[i % len(COLORS)]
            plot.traj(ax, plot.PlotMode.xy, traj, label=robot_name.replace('_', '-'),
                      color=color, plot_start_end_markers=True)

        ax.legend()
        ax.grid(True)

        # Save the figure
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f'combined_trajectory_2d.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved combined 2D trajectory plot to {output_dir}")

    else:
        # Plot individual trajectories
        for i, robot_name in enumerate(robot_names):
            traj = trajectories.get(robot_name)
            if traj is None:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_title(f"{robot_name} - {title}")
            ax.set_xlabel(r'$X$ [m]')
            ax.set_ylabel(r'$Y$ [m]')

            color = COLORS[i % len(COLORS)]
            plot.traj(ax, plot.PlotMode.xy, traj, label=robot_name.replace('_', '-'),
                      color=color, plot_start_end_markers=True)

            ax.legend()
            ax.grid(True)

            # Save the figure
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig.savefig(os.path.join(output_dir, f'{robot_name}_trajectory_2d.pdf'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved 2D trajectory plot for {robot_name}")


def plot_trajectories_3d(trajectories, robot_names, output_dir, title, combined=True):
    """Plot 3D trajectories, either combined or individually"""

    if combined:
        # Plot all trajectories on one figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        ax.set_xlabel(r'$X$ [m]')
        ax.set_ylabel(r'$Y$ [m]')
        ax.set_zlabel(r'$Z$ [m]')

        for i, robot_name in enumerate(robot_names):
            traj = trajectories.get(robot_name)
            if traj is None:
                continue

            color = COLORS[i % len(COLORS)]
            xs = traj.positions_xyz[:, 0]
            ys = traj.positions_xyz[:, 1]
            zs = traj.positions_xyz[:, 2]

            # Plot trajectory
            ax.plot(xs, ys, zs, label=robot_name.replace('_', '-'), color=color)

            # Mark start and end points
            ax.scatter(xs[0], ys[0], zs[0], color=color, marker='o', s=50)
            ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker='x', s=50)

        ax.legend()
        ax.grid(True)

        # Save the figure
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.savefig(os.path.join(output_dir, f'combined_trajectory_3d.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved combined 3D trajectory plot to {output_dir}")

    else:
        # Plot individual trajectories
        for i, robot_name in enumerate(robot_names):
            traj = trajectories.get(robot_name)
            if traj is None:
                continue

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title(f"{robot_name} - {title}")
            ax.set_xlabel(r'$X$ [m]')
            ax.set_ylabel(r'$Y$ [m]')
            ax.set_zlabel(r'$Z$ [m]')

            color = COLORS[i % len(COLORS)]
            xs = traj.positions_xyz[:, 0]
            ys = traj.positions_xyz[:, 1]
            zs = traj.positions_xyz[:, 2]

            # Plot trajectory
            ax.plot(xs, ys, zs, label=robot_name.replace('_', '-'), color=color)

            # Mark start and end points
            ax.scatter(xs[0], ys[0], zs[0], color=color, marker='o', s=50)
            ax.scatter(xs[-1], ys[-1], zs[-1], color=color, marker='x', s=50)

            ax.legend()
            ax.grid(True)

            # Save the figure
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            fig.savefig(os.path.join(output_dir, f'{robot_name}_trajectory_3d.pdf'),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved 3D trajectory plot for {robot_name}")


def format_timestamp(timestamp):
    """Format a timestamp in a human-readable way (if possible)"""
    try:
        # Try to format as datetime if it looks like a Unix timestamp
        # Check if it's a typical Unix timestamp (after year 2001)
        if timestamp > 1000000000:
            dt = datetime.datetime.fromtimestamp(timestamp)
            return f"{timestamp:.2f} ({dt.strftime('%Y-%m-%d %H:%M:%S')})"
        else:
            # Otherwise just return the number
            return f"{timestamp:.2f}"
    except:
        # If any error, just return the original timestamp
        return f"{timestamp:.2f}"


def print_trajectory_stats(trajectories, robot_names, start_time, end_time):
    """Print statistics about the trajectories"""
    print("\nTrajectory Statistics:")
    print("-" * 100)
    print(f"{'Robot':<12} {'Start Time':>25} {'End Time':>25} {'Duration':>12} {'Points':>8} {'Distance (m)':>15}")
    print("-" * 100)

    for robot_name in robot_names:
        traj = trajectories.get(robot_name)
        if traj is None:
            print(
                f"{robot_name:<12} {'N/A':>25} {'N/A':>25} {'N/A':>12} {'N/A':>8} {'N/A':>15}")
            continue

        # Calculate trajectory length
        positions = traj.positions_xyz
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        total_distance = np.sum(distances)

        # Get time information
        t_start = traj.timestamps[0]
        t_end = traj.timestamps[-1]
        duration = t_end - t_start

        print(f"{robot_name:<12} {format_timestamp(t_start):>25} {format_timestamp(t_end):>25} {duration:>12.2f} {len(traj.timestamps):>8} {total_distance:>15.3f}")

    print("-" * 100)
    if start_time is not None or end_time is not None:
        start_info = format_timestamp(start_time) if start_time else "start"
        end_info = format_timestamp(end_time) if end_time else "end"
        print(
            f"Note: Trajectories filtered by time range: {start_info} to {end_info}")
    print()


def main():
    args = parse_args()

    # Get list of robots to process
    selected_robots = args.robots

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load ground truth trajectories
    gt_trajectories = read_ground_truth(
        args.gt_dir, selected_robots, args.dataset)

    # Filter trajectories based on time range
    filtered_trajectories = {}
    for robot_name, traj in gt_trajectories.items():
        if traj is not None:
            filtered_traj = filter_trajectory_by_time(
                traj, args.start_time, args.end_time)
            filtered_trajectories[robot_name] = filtered_traj

    # Print trajectory statistics
    print_trajectory_stats(filtered_trajectories,
                           selected_robots, args.start_time, args.end_time)

    # Create title with time range if specified
    title = args.title
    if args.start_time is not None or args.end_time is not None:
        start_info = f"{args.start_time:.2f}" if args.start_time else "start"
        end_info = f"{args.end_time:.2f}" if args.end_time else "end"
        time_info = f" (Time: {start_info} to {end_info})"
        title += time_info

    # Generate plots
    if args.plot_3d:
        plot_trajectories_3d(filtered_trajectories, selected_robots, args.output_dir,
                             title, combined=not args.individual)
    else:
        plot_trajectories_2d(filtered_trajectories, selected_robots, args.output_dir,
                             title, combined=not args.individual)

    print(f"All plots saved to {args.output_dir}")


if __name__ == '__main__':
    main()
