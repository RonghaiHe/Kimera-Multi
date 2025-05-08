#!/usr/bin/env python3
'''
Copyright © 5, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-04-15
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-05-08 18:14:04
FilePath: /src/kimera_multi/evaluation/factor_statistics.py
Version: 1.0.0
Description: Analyze DPGO logs and pose constraints to generate statistics
'''

import os
import glob
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import scienceplots

# Define robot names
ROBOT_NAMES = ['sparkal2', 'acl_jackal2', 'acl_jackal']

'''
python factor_statistics.py \
    --base_dir /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_range_/exp_range2/log_data_12_08/
'''


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze DPGO logs and pose constraints statistics')
    parser.add_argument('--base_dir', type=str, default='/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_range_/exp_range2/log_data_12_08/',
                        help='Base directory containing robot subdirectories')
    parser.add_argument('--num_robots', type=int, default=3,
                        help='Number of robots to process')
    parser.add_argument('--output_dir', type=str, default='statistics_results',
                        help='Directory to save analysis results')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots of the statistics')
    parser.add_argument('--multi', action='store_true',
                        help='Flag to indicate multi-robot mode (read from distributed directory)')
    return parser.parse_args()


def find_dpgo_log_files(base_dir, robot_name):
    """Find all DPGO log files for a specific robot"""
    search_pattern = os.path.join(
        base_dir, robot_name, 'distributed', 'dpgo_log_*.csv')
    return glob.glob(search_pattern)


def find_latest_pose_constraint_file(base_dir, robot_name):
    """Find the pose constraint file with the largest number in the filename"""
    search_pattern = os.path.join(
        base_dir, robot_name, 'distributed', 'pose_constraint_*.csv')
    files = glob.glob(search_pattern)

    if not files:
        return None

    # Extract numbers from filenames and find the maximum
    def extract_number(filename):
        try:
            basename = os.path.basename(filename)
            number_part = basename.replace(
                'pose_constraint_', '').replace('.csv', '')
            return int(number_part)
        except ValueError:
            return -1

    files_with_numbers = [(f, extract_number(f)) for f in files]
    files_with_numbers.sort(key=lambda x: x[1], reverse=True)

    # Fix: Check if there are valid files before returning the first one
    if len(files_with_numbers) > 0:
        # Return the file with the highest number
        return files_with_numbers[1][0]
    else:
        return None


def find_lcd_result_file(base_dir, robot_name):
    """Find the LCD result file for a specific robot"""
    file_path = os.path.join(base_dir, robot_name,
                             'single', 'output_lcd_result.csv')
    if os.path.exists(file_path):
        return file_path
    return None


def process_lcd_results(base_dir, robot_names):
    """Process LCD result files to count loop closures"""
    stats = {
        'robot_name': [],
        'lcd_loops': []
    }

    print("\nProcessing LCD result files...")
    for robot_name in tqdm(robot_names, desc="Processing LCD results"):
        lcd_file = find_lcd_result_file(base_dir, robot_name)
        lcd_loops = 0

        if lcd_file:
            try:
                print(f"Found LCD file for {robot_name}: {lcd_file}")

                # First check file content manually to debug
                with open(lcd_file, 'r') as f:
                    first_lines = [f.readline().strip()
                                   for _ in range(5) if f.readline()]
                    print(f"First few lines of LCD file for {robot_name}:")
                    for i, line in enumerate(first_lines[:3]):
                        print(f"  Line {i}: {line[:100]}...")

                # Based on the file inspection, we can see it's comma-separated
                # Try reading as CSV with comma separator
                try:
                    # Read with comma separator - skip the header if present
                    df = pd.read_csv(lcd_file, sep=',', comment='#')
                    print(
                        f"Successfully loaded LCD file with columns: {df.columns.tolist()}")

                    # Locate the isLoop column (4th column)
                    if 'isLoop' in df.columns:
                        lcd_loops = int((df['isLoop'] == 1).sum())
                    elif len(df.columns) >= 4:
                        # If column name is not 'isLoop', try 4th column (0-based index 3)
                        isloop_col = df.columns[3]
                        lcd_loops = int((df[isloop_col] == 1).sum())

                    print(
                        f"Found {lcd_loops} loops in LCD file for {robot_name}")

                except Exception as e:
                    print(f"Error parsing LCD file with comma separator: {e}")

                    # Try one more approach - manual line parsing
                    try:
                        lcd_loops = 0
                        with open(lcd_file, 'r') as f:
                            for line in f:
                                # Skip header/comment lines
                                if line.startswith('#'):
                                    continue
                                parts = line.strip().split(',')
                                if len(parts) >= 4:  # Make sure we have enough columns
                                    try:
                                        is_loop = int(parts[3])
                                        if is_loop == 1:
                                            lcd_loops += 1
                                    except (ValueError, IndexError):
                                        continue
                        print(
                            f"Manual parsing found {lcd_loops} loops for {robot_name}")
                    except Exception as manual_e:
                        print(f"Manual parsing also failed: {manual_e}")

            except Exception as e:
                print(f"Error processing LCD file for {robot_name}: {e}")
        else:
            print(f"No LCD result file found for {robot_name}")

        stats['robot_name'].append(robot_name)
        stats['lcd_loops'].append(lcd_loops)

    return pd.DataFrame(stats)


# Enhance process_dpgo_logs function to better report iteration statistics
def process_dpgo_logs(base_dir, robot_names, is_multi=True):
    """Process DPGO log files to calculate average iteration time and final iteration counts"""
    stats = {
        'robot_name': [],
        'final_iteration': [],        # Final iteration count
        'avg_iteration': [],          # Average iteration across files
        'total_files': [],
        # Average time for all iterations (calculated from final iterations)
        'avg_iter_time': [],
        'min_iter_time': [],
        'max_iter_time': [],
        # Average time per single iteration (from iter_time_sec column)
        'mean_iter_time_sec': [],
        'total_iterations': []        # Total number of iterations across all files
    }

    # If not multi-robot mode, return empty stats
    if not is_multi:
        # Create empty dataframe with same structure
        for robot_name in robot_names:
            stats['robot_name'].append(robot_name)
            stats['final_iteration'].append(0)
            stats['avg_iteration'].append(0)
            stats['total_files'].append(0)
            stats['avg_iter_time'].append(0)
            stats['min_iter_time'].append(0)
            stats['max_iter_time'].append(0)
            # Add default value for the new field
            stats['mean_iter_time_sec'].append(0)
            # Add default value for total iterations
            stats['total_iterations'].append(0)
        return pd.DataFrame(stats)

    # Track acl_jackal's final iteration separately
    acl_jackal_final_iter = None
    acl_jackal_all_iterations = []  # Store all iterations for averaging
    # Store iterations + len(robot_names)-1
    acl_jackal_adjusted_iterations = []

    # New: Track successful final iterations for time averaging
    final_iterations_only = []

    # New: Track all iter_time_sec values
    all_iter_time_sec = []

    print("\nProcessing DPGO log files...")
    for robot_name in tqdm(robot_names, desc="Processing robots"):
        dpgo_files = find_dpgo_log_files(base_dir, robot_name)

        if not dpgo_files:
            print(f"No DPGO log files found for {robot_name}")
            stats['robot_name'].append(robot_name)
            stats['final_iteration'].append(0)
            stats['avg_iteration'].append(0)
            stats['total_files'].append(0)
            stats['avg_iter_time'].append(0)
            stats['min_iter_time'].append(0)
            stats['max_iter_time'].append(0)
            # Add default value for the new field
            stats['mean_iter_time_sec'].append(0)
            # Add default value for total iterations
            stats['total_iterations'].append(0)
            continue

        robot_final_iterations = []
        robot_iter_times = []
        robot_total_files = len(dpgo_files)

        # We only need to process the first robot's logs for time calculation
        if robot_name == ROBOT_NAMES[0]:
            print(
                f"Processing {len(dpgo_files)} DPGO log files for {robot_name}...")

            # Debug: Print out one file to examine structure
            if dpgo_files:
                print(
                    f"Examining structure of first DPGO log file: {dpgo_files[0]}")
                try:
                    with open(dpgo_files[0], 'r') as f:
                        header = f.readline().strip()
                        print(f"Header: {header}")
                        # Print a few sample lines
                        print("Sample lines:")
                        for _ in range(3):
                            line = f.readline().strip()
                            print(f"  {line}")
                except Exception as e:
                    print(f"Error reading sample file: {e}")

            for file_path in dpgo_files:
                try:
                    # Check if the file is empty
                    if os.path.getsize(file_path) == 0:
                        print(f"Skipping empty file: {file_path}")
                        continue

                    # New: Improved method to extract iter_time_sec values
                    # Based on the sample file format: robot_id, cluster_id, num_active_robots, iteration, num_poses, bytes_received, iter_time_sec, ...
                    with open(file_path, 'r') as f:
                        lines = f.readlines()

                    # Extract header to identify column positions
                    header_line = lines[0] if lines else ""
                    header_parts = [p.strip() for p in header_line.split(',')]

                    # Find the index of iter_time_sec in the header
                    iter_time_idx = -1
                    for i, part in enumerate(header_parts):
                        if 'iter_time_sec' in part:
                            iter_time_idx = i
                            break

                    if iter_time_idx == -1:
                        print(
                            f"Cannot find iter_time_sec in header: {header_line}")
                        # Try default position (6th column, index 5 in 0-based indexing)
                        iter_time_idx = 6
                        print(
                            f"Using default position {iter_time_idx} for iter_time_sec")

                    # Extract iter_time_sec values from each data line
                    local_iter_times = []
                    for i, line in enumerate(lines[1:], 1):
                        # Skip non-data lines
                        if not line.strip() or 'UPDATE_WEIGHT' in line or 'TERMINATE' in line:
                            continue

                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) > iter_time_idx:
                            try:
                                iter_time = float(parts[iter_time_idx])
                                local_iter_times.append(iter_time)
                            except (ValueError, IndexError) as e:
                                if i < 5:  # Only print errors for the first few lines
                                    print(
                                        f"Error extracting iter_time_sec from line {i}: {e}")
                                    print(f"Line content: {line}")
                                    print(f"Parsed parts: {parts}")

                    # Validate and add to global list
                    if local_iter_times:
                        print(
                            f"Found {len(local_iter_times)} iter_time_sec values in {file_path}")
                        print(f"Sample values: {local_iter_times[:5]}")
                        print(
                            f"Min: {min(local_iter_times)}, Max: {max(local_iter_times)}, Mean: {sum(local_iter_times)/len(local_iter_times):.4f}")
                        all_iter_time_sec.extend(local_iter_times)

                    # ...existing code for loading the file and finding final iteration...

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

            # Now process the full files for other statistics
            for file_path in dpgo_files:
                try:
                    # Check if the file is empty
                    if os.path.getsize(file_path) == 0:
                        continue

                    # First read file content to detect format
                    with open(file_path, 'r') as f:
                        first_lines = [f.readline().strip()
                                       for _ in range(3) if f.readline()]

                    # Try multiple approaches to parse the file
                    iter_values = []

                    # Try common separators
                    for sep in [',', ' ', '\t']:
                        try:
                            # Try reading with this separator
                            df = pd.read_csv(
                                file_path, sep=sep, on_bad_lines='skip')

                            if len(df.columns) <= 1:
                                continue

                            # Look for iteration columns
                            iter_cols = [col for col in df.columns if 'iter' in col.lower(
                            ) and not 'time' in col.lower()]

                            # Process iteration number data
                            if iter_cols:
                                for iter_col in iter_cols:
                                    try:
                                        df[iter_col] = pd.to_numeric(
                                            df[iter_col], errors='coerce')
                                        valid_iters = df[iter_col].dropna()
                                        if not valid_iters.empty:
                                            # Get all iteration values for averaging
                                            iter_values.extend(
                                                valid_iters.tolist())

                                            # Get max iteration value
                                            final_iter = valid_iters.max()
                                            robot_final_iterations.append(
                                                final_iter)

                                            # Store adjusted values (iter + num_robots-1)
                                            robot_adjustment = len(
                                                robot_names) - 1
                                            adjusted_values = [
                                                val + robot_adjustment for val in valid_iters.tolist()]
                                            acl_jackal_adjusted_iterations.extend(
                                                adjusted_values)
                                    except Exception:
                                        pass

                            # If we found useful data, no need to try more separators
                            if iter_values:
                                break

                        except Exception:
                            continue

                    # For acl_jackal, track all iterations for averaging
                    if iter_values:
                        # Store all iterations for later averaging
                        acl_jackal_all_iterations.extend(iter_values)
                        # For final iteration, add number of robots - 1
                        max_iter = max(iter_values)
                        robot_final_iterations.append(
                            max_iter + len(robot_names) - 1)

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

        # For other robots, just add placeholder stats
        else:
            stats['robot_name'].append(robot_name)
            stats['total_files'].append(robot_total_files)
            stats['final_iteration'].append(-1)
            stats['avg_iteration'].append(-1)
            stats['avg_iter_time'].append(0)
            stats['min_iter_time'].append(0)
            stats['max_iter_time'].append(0)
            stats['mean_iter_time_sec'].append(0)
            # Add default value for total iterations
            stats['total_iterations'].append(0)
            continue

        # Store robot information
        stats['robot_name'].append(robot_name)
        stats['total_files'].append(robot_total_files)

        # Calculate time statistics for first robot only
        if robot_name == ROBOT_NAMES[0]:
            # Handle final iteration count
            if robot_final_iterations:
                final_iter = max(robot_final_iterations)
                stats['final_iteration'].append(final_iter)
                acl_jackal_final_iter = final_iter
            else:
                stats['final_iteration'].append(0)

            # Calculate average iteration for first robot
            if acl_jackal_all_iterations:
                avg_iter = np.mean(acl_jackal_all_iterations)
                stats['avg_iteration'].append(avg_iter)
            else:
                stats['avg_iteration'].append(0)

            # Calculate iteration time as the average of final iterations
            if final_iterations_only:
                avg_iter_time = np.mean(final_iterations_only)
                min_iter_time = np.min(final_iterations_only)
                max_iter_time = np.max(final_iterations_only)

                stats['avg_iter_time'].append(avg_iter_time)
                stats['min_iter_time'].append(min_iter_time)
                stats['max_iter_time'].append(max_iter_time)

                # Print detailed calculation information
                print(f"\nDPGO Iteration Time Calculation:")
                print(
                    f"  Using final iterations from {len(final_iterations_only)} files")
                print(f"  Final iterations used: {final_iterations_only}")
                print(
                    f"  Sum: {sum(final_iterations_only)}, Count: {len(final_iterations_only)}")
                print(f"  Average = {avg_iter_time:.2f}")
                print(f"  Min = {min_iter_time}, Max = {max_iter_time}")
            else:
                stats['avg_iter_time'].append(0)
                stats['min_iter_time'].append(0)
                stats['max_iter_time'].append(0)
                print("No final iterations found for time calculation")

            # Calculate mean of iter_time_sec values
            if all_iter_time_sec:
                # Filter out any potential outliers or erroneous readings
                # Reasonable bounds for iteration time in seconds
                filtered_times = [t for t in all_iter_time_sec if 0 < t < 10]
                if len(filtered_times) != len(all_iter_time_sec):
                    print(
                        f"Filtered out {len(all_iter_time_sec) - len(filtered_times)} outlier values")

                if filtered_times:
                    mean_iter_time_sec = np.mean(filtered_times)
                    stats['mean_iter_time_sec'].append(mean_iter_time_sec)

                    # Print detailed calculation information for iter_time_sec
                    print(f"\nMean Iteration Time (iter_time_sec) Calculation:")
                    print(
                        f"  Using {len(filtered_times)} iter_time_sec values (after filtering)")
                    print(
                        f"  Sum: {sum(filtered_times):.4f}, Count: {len(filtered_times)}")
                    print(f"  Mean = {mean_iter_time_sec:.4f} seconds")
                    print(
                        f"  Min = {min(filtered_times):.4f}, Max = {max(filtered_times):.4f}")
                else:
                    stats['mean_iter_time_sec'].append(0)
                    print("No valid iter_time_sec values after filtering")
            else:
                stats['mean_iter_time_sec'].append(0)
                print("No iter_time_sec values found for mean calculation")

        # Store total iteration count
        if acl_jackal_all_iterations:
            stats['total_iterations'].append(len(acl_jackal_all_iterations))
        else:
            stats['total_iterations'].append(0)

    # Create the dataframe
    df_stats = pd.DataFrame(stats)

    # Update other robots with acl_jackal's final iteration for consistency
    if acl_jackal_final_iter is not None:
        for i, robot_name in enumerate(df_stats['robot_name']):
            if robot_name != ROBOT_NAMES[0]:
                df_stats.at[i, 'final_iteration'] = acl_jackal_final_iter

    return df_stats


def process_pose_constraints(base_dir, robot_names, is_multi=True, lcd_stats=None):
    """Process pose constraint files to count accepted loop closures"""
    stats = {
        'robot_name': [],
        'pri_lc': [],
        'pub_lc': [],
        'uwb': [],
        'total_ac': [],
        'und_pri_lc': [],
        'und_pub_lc': [],
        'und_uwb': [],
        'und': [],
        'ac_and_und': []
    }

    print("\nProcessing pose constraint files...")
    for robot_name in tqdm(robot_names, desc="Processing robots"):
        # Initialize loop closure counts
        pri_lc = 0
        pub_lc = 0
        uwb = 0
        undecided_private = 0
        undecided_public = 0
        undecided_uwb = 0

        # If we have LCD stats, use them for private loop closures
        if lcd_stats is not None:
            robot_lcd = lcd_stats[lcd_stats['robot_name'] == robot_name]
            if not robot_lcd.empty:
                pri_lc = robot_lcd.iloc[0]['lcd_loops']
                print(
                    f"Using LCD loop count ({pri_lc}) for {robot_name} private loop closures")

        # If in multi-robot mode, process distributed pose constraints as well
        if is_multi:
            pose_constraint_file = find_latest_pose_constraint_file(
                base_dir, robot_name)

            if pose_constraint_file:
                print(
                    f"Found pose constraint file for {robot_name}: {pose_constraint_file}")
                try:
                    # First check the file content to aid debugging
                    with open(pose_constraint_file, 'r') as f:
                        header = f.readline().strip()
                        first_line = f.readline().strip() if f.readable() else "No data"

                    print(f"Pose constraint file header: {header}")
                    print(f"First data line: {first_line}")

                    # Now try to read the CSV with more flexible options
                    df = pd.read_csv(pose_constraint_file)
                    print(
                        f"Successfully loaded CSV with {len(df)} rows and columns: {df.columns.tolist()}")

                    # Print first few rows for debugging
                    if not df.empty:
                        print(f"First row data: {df.iloc[0].to_dict()}")

                    # Count accepted loop closures with more careful column checking
                    accept_columns = [
                        col for col in df.columns if col.startswith('accept_')]
                    undecided_columns = [
                        col for col in df.columns if col.startswith('undecided_')]

                    print(f"Found acceptance columns: {accept_columns}")
                    print(f"Found undecided columns: {undecided_columns}")

                    # Convert columns to numeric first
                    for col in df.columns:
                        if col.startswith('accept_') or col.startswith('undecided_'):
                            # Try converting to numeric, coerce errors to NaN
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            # Replace NaN with 0 to safely sum
                            df[col] = df[col].fillna(0)
                            # Convert to int
                            df[col] = df[col].astype(int)

                    # Print data types to verify conversion
                    print(f"Column data types: {df.dtypes}")

                    # Count with better error handling
                    if 'accept_private_lc' in df.columns and lcd_stats is None:
                        try:
                            pri_lc = int(df['accept_private_lc'].sum())
                            print(f"Private LC count: {pri_lc}")
                        except Exception as e:
                            print(f"Error summing 'accept_private_lc': {e}")
                            print(
                                f"Sample values: {df['accept_private_lc'].head()}")

                    if 'accept_public_lc' in df.columns:
                        try:
                            pub_lc = int(df['accept_public_lc'].sum())
                            print(f"Public LC count: {pub_lc}")
                        except Exception as e:
                            print(f"Error summing 'accept_public_lc': {e}")

                    if 'accept_uwb' in df.columns:
                        try:
                            uwb = int(df['accept_uwb'].sum())
                            print(f"UWB count: {uwb}")
                        except Exception as e:
                            print(f"Error summing 'accept_uwb': {e}")

                    if 'undecided_private_lc' in df.columns:
                        try:
                            undecided_private = int(
                                df['undecided_private_lc'].sum())
                            print(
                                f"Undecided private LC count: {undecided_private}")
                        except Exception as e:
                            print(f"Error summing 'undecided_private_lc': {e}")

                    if 'undecided_public_lc' in df.columns:
                        try:
                            undecided_public = int(
                                df['undecided_public_lc'].sum())
                            print(
                                f"Undecided public LC count: {undecided_public}")
                        except Exception as e:
                            print(f"Error summing 'undecided_public_lc': {e}")
                    print(df.columns)
                    if 'undecided_uwb ' in df.columns:
                        try:
                            undecided_uwb = int(df['undecided_uwb '].sum())
                            print(f"Undecided UWB count: {undecided_uwb}")
                        except Exception as e:
                            print(f"Error summing 'undecided_uwb': {e}")

                except Exception as e:
                    print(f"Error processing {pose_constraint_file}: {e}")
                    # Add debug info
                    try:
                        import os
                        file_size = os.path.getsize(pose_constraint_file)
                        print(f"File size: {file_size} bytes")
                    except:
                        print("Could not get file info")
            else:
                print(f"No pose constraint files found for {robot_name}")

        # Total undecided factors
        total_undecided = undecided_private + undecided_public + undecided_uwb

        # Total accepted factors
        total_ac = pri_lc + pub_lc + uwb

        # Total factors including undecided
        ac_and_und = total_ac + total_undecided

        stats['robot_name'].append(robot_name)
        stats['pri_lc'].append(pri_lc)
        stats['pub_lc'].append(pub_lc)
        stats['uwb'].append(uwb)
        stats['total_ac'].append(total_ac)
        stats['und'].append(total_undecided)
        stats['ac_and_und'].append(ac_and_und)
        # Add individual undecided components
        stats['und_pri_lc'].append(undecided_private)
        stats['und_pub_lc'].append(undecided_public)
        stats['und_uwb'].append(undecided_uwb)

        print(
            f"Summary for {robot_name}: Ac={total_ac}, Un={total_undecided}, Total={ac_and_und}")

    result_df = pd.DataFrame(stats)
    print(f"Final constraint statistics dataframe: {len(result_df)} rows")
    return result_df


# Enhance visualization to include both iteration count and time metrics
def visualize_statistics(dpgo_stats, constraint_stats, output_dir, is_multi=True):
    """Generate visualization plots for the statistics"""
    plt.style.use(['science', 'ieee', 'no-latex'])

    # Only plot DPGO statistics in multi-robot mode
    if is_multi:
        # Plot average iteration time (time for all iterations)
        plt.figure(figsize=(10, 6))
        plt.bar(dpgo_stats['robot_name'], dpgo_stats['avg_iter_time'])
        plt.xlabel('Robot')
        plt.ylabel('Average Iteration Time (seconds)')
        plt.title('DPGO Average Total Optimization Time by Robot')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, 'dpgo_avg_iter_time.pdf'), dpi=300)
        plt.close()

        # Plot mean time per iteration (iter_time_sec)
        if 'mean_iter_time_sec' in dpgo_stats.columns:
            plt.figure(figsize=(10, 6))
            acl_jackal_idx = dpgo_stats[dpgo_stats['robot_name']
                                        == ROBOT_NAMES[0]].index
            if not acl_jackal_idx.empty:
                acl_jackal_data = dpgo_stats.iloc[acl_jackal_idx[0]]
                plt.bar(['acl_jackal'], [acl_jackal_data['mean_iter_time_sec']])
                plt.xlabel('Robot')
                plt.ylabel('Mean Time per Iteration (seconds)')
                plt.title('Mean Time per Optimization Step (iter_time_sec)')
                plt.tight_layout()
                plt.savefig(os.path.join(
                    output_dir, 'mean_iter_time_sec.pdf'), dpi=300)
            plt.close()

        # Plot average iteration count
        plt.figure(figsize=(10, 6))
        acl_jackal_idx = dpgo_stats[dpgo_stats['robot_name']
                                    == ROBOT_NAMES[0]].index
        if not acl_jackal_idx.empty:
            acl_jackal_data = dpgo_stats.iloc[acl_jackal_idx[0]]
            plt.bar(['acl_jackal'], [acl_jackal_data['avg_iteration']])
            plt.xlabel('Robot')
            plt.ylabel('Average Iteration Count')
            plt.title('DPGO Average Iteration Count (acl_jackal)')
            plt.tight_layout()
            plt.savefig(os.path.join(
                output_dir, 'dpgo_avg_iteration.pdf'), dpi=300)
        plt.close()

        # Plot final iteration count
        plt.figure(figsize=(10, 6))
        acl_jackal_idx = dpgo_stats[dpgo_stats['robot_name']
                                    == ROBOT_NAMES[0]].index
        if not acl_jackal_idx.empty:
            acl_jackal_data = dpgo_stats.iloc[acl_jackal_idx[0]]
            plt.bar(['acl_jackal'], [acl_jackal_data['final_iteration']])
            plt.xlabel('Robot')
            plt.ylabel('Final Iteration Count')
            plt.title('DPGO Final Iteration Count (acl_jackal)')
            plt.tight_layout()
            plt.savefig(os.path.join(
                output_dir, 'dpgo_final_iteration.pdf'), dpi=300)
        plt.close()

        # Add a new combined plot for iteration stats
        plt.figure(figsize=(12, 8))
        acl_jackal_idx = dpgo_stats[dpgo_stats['robot_name']
                                    == ROBOT_NAMES[0]].index
        if not acl_jackal_idx.empty:
            acl_jackal_data = dpgo_stats.iloc[acl_jackal_idx[0]]

            # Create a grouped bar chart
            labels = ['Average Iteration',
                      'Final Iteration', 'Total Iterations']
            values = [acl_jackal_data['avg_iteration'],
                      acl_jackal_data['final_iteration'],
                      acl_jackal_data['total_iterations']]

            x = np.arange(len(labels))
            plt.bar(x, values)
            plt.xlabel('Metric')
            plt.ylabel('Count')
            plt.title('DPGO Iteration Statistics (acl_jackal)')
            plt.xticks(x, labels)
            plt.tight_layout()
            plt.savefig(os.path.join(
                output_dir, 'dpgo_iteration_stats.pdf'), dpi=300)
        plt.close()

    # Plot accepted loop closures (always plot these)
    plt.figure(figsize=(12, 8))
    width = 0.15
    x = np.arange(len(constraint_stats['robot_name']))
    plt.bar(x - width*1.5,
            constraint_stats['pri_lc'], width, label='Private LC')
    plt.bar(x - width*0.5,
            constraint_stats['pub_lc'], width, label='Public LC')
    plt.bar(x + width*0.5, constraint_stats['uwb'], width, label='UWB')
    plt.bar(x + width*1.5,
            constraint_stats['und'], width, label='Undecided')
    plt.xlabel('Robot')
    plt.ylabel('Count')
    plt.title('Constraints by Type and Robot')
    plt.xticks(x, constraint_stats['robot_name'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_constraints.pdf'), dpi=300)
    plt.close()

    # Plot stacked bar chart of total constraints with undecided highlighted
    plt.figure(figsize=(12, 8))
    x = np.arange(len(constraint_stats['robot_name']))
    plt.bar(x, constraint_stats['total_ac'], label='Accepted')
    plt.bar(x, constraint_stats['und'], bottom=constraint_stats['total_ac'],
            label='Undecided', alpha=0.7)
    plt.xlabel('Robot')
    plt.ylabel('Count')
    plt.title('Total Constraints by Robot (Accepted + Undecided)')
    plt.xticks(x, constraint_stats['robot_name'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ac_and_und.pdf'), dpi=300)
    plt.close()

    # Add a new plot for undecided factors breakdown
    plt.figure(figsize=(12, 8))
    width = 0.2
    x = np.arange(len(constraint_stats['robot_name']))
    plt.bar(x - width, constraint_stats['und_pri_lc'],
            width, label='Undecided Private LC')
    plt.bar(x, constraint_stats['und_pub_lc'],
            width, label='Undecided Public LC')
    plt.bar(x + width, constraint_stats['und_uwb'],
            width, label='Undecided UWB')

    plt.xlabel('Robot')
    plt.ylabel('Count')
    plt.title('Undecided Constraints Breakdown by Robot')
    plt.xticks(x, constraint_stats['robot_name'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'undecided_breakdown.pdf'), dpi=300)
    plt.close()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get robot names to process
    robot_names = ROBOT_NAMES[:args.num_robots]
    print(f"Processing data for robots: {', '.join(robot_names)}")
    print(f"Multi-robot mode: {args.multi}")

    # Process LCD results first (for both single and multi modes)
    lcd_stats = process_lcd_results(args.base_dir, robot_names)

    # Only process DPGO logs in multi-robot mode
    dpgo_stats = process_dpgo_logs(
        args.base_dir, robot_names, is_multi=args.multi)

    # Process pose constraints with LCD stats to override private loop closures
    print("\nStarting constraint file processing...")
    constraint_stats = process_pose_constraints(
        args.base_dir, robot_names, is_multi=args.multi, lcd_stats=lcd_stats)
    print(f"Constraint statistics shape: {constraint_stats.shape}")

    # Save statistics to CSV files
    dpgo_stats.to_csv(os.path.join(
        args.output_dir, 'dpgo_statistics.csv'), index=False)
    constraint_stats.to_csv(os.path.join(
        args.output_dir, 'constraint_statistics.csv'), index=False)
    lcd_stats.to_csv(os.path.join(
        args.output_dir, 'lcd_statistics.csv'), index=False)

    # Print summary
    print("\nLCD Statistics Summary:")
    print(lcd_stats)

    if args.multi:
        print("\nDPGO Statistics Summary:")
        print(dpgo_stats)

        # Enhanced reporting of iteration statistics
        acl_jackal_data = dpgo_stats[dpgo_stats['robot_name']
                                     == ROBOT_NAMES[0]]
        if not acl_jackal_data.empty:
            acl_data = acl_jackal_data.iloc[0]
            print("\nDetailed Iteration Statistics for", ROBOT_NAMES[0])
            print(
                f"  Average Iteration Count: {acl_data['avg_iteration']:.2f}")
            print(f"  Final Iteration Count: {acl_data['final_iteration']}")
            print(
                f"  Total Iterations Processed: {acl_data['total_iterations']}")
            print(
                f"  Mean Time per Iteration Step: {acl_data['mean_iter_time_sec']:.4f} seconds")
            print(
                f"  Average Total Optimization Time: {acl_data['avg_iter_time']:.4f} seconds")

    print("\nConstraint Statistics Summary:")
    print(constraint_stats)

    # Calculate overall averages
    overall_avg_iter_time = dpgo_stats['avg_iter_time'].mean(
    ) if 'avg_iter_time' in dpgo_stats.columns else 0
    overall_mean_iter_time_sec = dpgo_stats['mean_iter_time_sec'].mean(
    ) if 'mean_iter_time_sec' in dpgo_stats.columns else 0
    overall_avg_iteration = dpgo_stats.loc[dpgo_stats['robot_name'] == ROBOT_NAMES[0],
                                           'avg_iteration'].values[0] if 'avg_iteration' in dpgo_stats.columns else 0
    acl_jackal_final_iteration = dpgo_stats.loc[dpgo_stats['robot_name']
                                                == ROBOT_NAMES[0], 'final_iteration'].values if 'final_iteration' in dpgo_stats.columns else []
    acl_jackal_iteration = acl_jackal_final_iteration[0] if len(
        acl_jackal_final_iteration) > 0 else None
    total_ac = constraint_stats['total_ac'].sum()
    ac_and_und = constraint_stats['ac_and_und'].sum()
    total_lcd_loops = lcd_stats['lcd_loops'].sum(
    ) if 'lcd_loops' in lcd_stats.columns else 0

    ac_and_und_pri_lc = constraint_stats['und_pri_lc'].sum(
    ) + constraint_stats['pri_lc'].sum()
    total_lc = (ac_and_und - ac_and_und_pri_lc + 1) // 2 + ac_and_und_pri_lc

    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Total LCD loops across all robots: {total_lcd_loops}")

    if args.multi:
        print(f"Average iteration count: {overall_avg_iteration:.2f}")
        print(
            f"Average total optimization time across all robots: {overall_avg_iter_time:.4f} seconds")
        print(
            f"Mean time per iteration step across all robots: {overall_mean_iter_time_sec:.4f} seconds")
        print(f"acl_jackal final iteration count: {acl_jackal_iteration}")

    print(f"Total accepted constraints: {total_ac}")
    print(f"Total loop closures including undecided: {total_lc}")

    # Generate visualizations if requested
    if args.visualize:
        print("\nGenerating visualization plots...")
        visualize_statistics(dpgo_stats, constraint_stats,
                             args.output_dir, is_multi=args.multi)

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
