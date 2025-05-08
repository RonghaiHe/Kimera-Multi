#!/usr/bin/env python3
'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-03-14 12:30:00
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-04-18 11:04:51
FilePath: /src/kimera_multi/evaluation/uwb_evaluation.py
Version: 1.0.0
Description: Evaluate UWB-based relative pose accuracy by comparing with ground truth
'''

import scienceplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive

plt.style.use(['science', 'ieee', 'no-latex'])

# Define robot ID mapping
ID2ROBOT = [
    'sparkal2',
    'acl_jackal2',
    'acl_jackal',
    'sparkal1',
    'hathor',
    'thoth',
    'apis',
    'sobek'
]

DATE2DATASET = {'1207': 'campus_tunnels_12_07',
                '1014': 'campus_outdoor_10_14',
                '1208': 'campus_hybrid_12_08'}

'''
python uwb_evaluation.py \
    --uwb_log_base_path /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/paper/campus_hybrid_12_08/exp_range_/exp_range1/log_data_12_08/
'''

t_body_uwb = np.array([[-0.0702200000733137, 0.04240000000223517, 0.0210199999809265],
                       [-0.0082200000733137, -
                           0.011599999997764829, -0.0349800000190735],
                       [0.1107799999266863, 0.04740000000223517, -0.051980000019073505]])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate UWB-based relative pose estimation')
    parser.add_argument('--date', type=str, default='1208',
                        choices=list(DATE2DATASET.keys()),
                        help='Date of the dataset (e.g., 1207, 1014, 1208)')
    parser.add_argument('--gt_basic_path', type=str,
                        default='/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets',
                        help='Base path to the GT of multi-robot datasets')
    parser.add_argument('--uwb_log_base_path', type=str,
                        default='/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/campus_hybrid_12_08/log_data_12_08/',
                        help='Base path to the distributed UWB log directory')
    parser.add_argument('--num_robots', type=int, default=3,
                        help='Number of robots in the dataset, often 6 or 8')
    parser.add_argument('--output_dir', type=str,
                        default='uwb_evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--tolerance', type=float, default=0.1,
                        help='Time tolerance for matching timestamps (seconds)')
    parser.add_argument('--submap_csv_path', type=str,
                        default=None,
                        help='Path to kimera_distributed_submaps.csv file')
    return parser.parse_args()


def read_groundtruth_tum(file_path):
    """Read the ground truth poses from a TUM format file"""
    try:
        groundtruth_data = pd.read_csv(file_path, sep=' ')
        groundtruth_data.columns = ['timestamp',
                                    'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
        return groundtruth_data
    except Exception as e:
        print(f"Error reading ground truth file {file_path}: {e}")
        return None


def find_closest_pose(timestamp, groundtruth_data, tolerance=0.1):
    """
    Find the closest pose in groundtruth_data within the given time tolerance

    Args:
        timestamp: UWB timestamp in seconds
        groundtruth_data: DataFrame with ground truth poses (timestamp in nanoseconds)
        tolerance: Time tolerance in seconds

    Returns:
        Closest pose or None if no match found within tolerance
    """
    if groundtruth_data is None or groundtruth_data.empty:
        return None

    closest_pose = None
    min_diff = float('inf')

    # Convert tolerance to nanoseconds for comparison
    tolerance_ns = tolerance * 1e9

    for _, row in groundtruth_data.iterrows():
        # Convert timestamp (in seconds) to nanoseconds for comparison with GT timestamp
        diff = abs(row['timestamp'] - timestamp)
        if diff < min_diff and diff <= tolerance:
            min_diff = diff
            closest_pose = row[1:].values

    return closest_pose


def pose_to_transformation_matrix(pose):
    """Convert pose [tx,ty,tz,qx,qy,qz,qw] to 4x4 transformation matrix"""
    if pose is None:
        return None

    # Extract translation and rotation
    tx, ty, tz = pose[0:3]
    qx, qy, qz, qw = pose[3:7]

    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # Create homogeneous transformation matrix
    transform = np.eye(4)
    transform[0:3, 0:3] = rot_matrix
    transform[0:3, 3] = [tx, ty, tz]

    return transform


def calculate_pose_error(T_gt_i, T_gt_j, T_uwb_ij):
    """
    Calculate the error between ground truth and UWB-based relative pose
    Computes: T_j⁻¹ * T_i * T_ij - I
    """
    if T_gt_i is None or T_gt_j is None or T_uwb_ij is None:
        return None, None, None

    error_matrix = T_gt_j - T_gt_i @ T_uwb_ij

    # Extract rotation error
    rot_error_matrix = error_matrix[:3, :3]
    rot_error = np.linalg.norm(rot_error_matrix, 'fro')  # Frobenius norm

    # Extract translation error
    trans_error = np.linalg.norm(error_matrix[:3, 3])

    return error_matrix, rot_error, trans_error


def uwb_transformation_matrix(uwb_data):
    """Convert UWB data (qx,qy,qz,qw,tx,ty,tz) to 4x4 transformation matrix"""
    try:
        qx, qy, qz, qw = uwb_data['qx'], uwb_data['qy'], uwb_data['qz'], uwb_data['qw']
        tx, ty, tz = uwb_data['tx'], uwb_data['ty'], uwb_data['tz']

        rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

        transform = np.eye(4)
        transform[0:3, 0:3] = rot_matrix
        transform[0:3, 3] = [tx, ty, tz]

        # Create Kalman filter transformation matrix
        qx_kf, qy_kf, qz_kf, qw_kf = uwb_data['qx_kf'], uwb_data['qy_kf'], uwb_data['qz_kf'], uwb_data['qw_kf']
        tx_kf, ty_kf, tz_kf = uwb_data['tx_kf'], uwb_data['ty_kf'], uwb_data['tz_kf']

        rot_matrix_kf = R.from_quat([qx_kf, qy_kf, qz_kf, qw_kf]).as_matrix()

        transform_kf = np.eye(4)
        transform_kf[0:3, 0:3] = rot_matrix_kf
        transform_kf[0:3, 3] = [tx_kf, ty_kf, tz_kf]

        # Create distance-based transformation matrix
        qx_dis, qy_dis, qz_dis, qw_dis = uwb_data['qx_dis'], uwb_data[
            'qy_dis'], uwb_data['qz_dis'], uwb_data['qw_dis']
        tx_dis, ty_dis, tz_dis = uwb_data['tx_dis'], uwb_data['ty_dis'], uwb_data['tz_dis']

        # Handle potential string-to-float conversion issues
        try:
            rot_matrix_dis = R.from_quat([float(qx_dis), float(
                qy_dis), float(qz_dis), float(qw_dis)]).as_matrix()
            transform_dis = np.eye(4)
            transform_dis[0:3, 0:3] = rot_matrix_dis
            transform_dis[0:3, 3] = [
                float(tx_dis), float(ty_dis), float(tz_dis)]
        except ValueError as e:
            print(f"Warning: Error in distance-based transform: {e}")
            transform_dis = np.eye(4)  # Return identity matrix as fallback

        # Calculate average UWB distance from available measurements
        uwb_distances = []
        uwb_dis_all = [-1]*9
        residual = []
        residual2 = []
        for i in range(9):  # Check distance measurements (dis0 through dis8)
            distance_key = f'dis{i}'
            if distance_key in uwb_data and not np.isnan(uwb_data[distance_key]):
                # Try to convert to float in case it's a string
                try:
                    distance = float(uwb_data[distance_key])
                    if distance >= 0:
                        uwb_distances.append(distance)
                        uwb_dis_all[i] = distance

                        idx_i = i // 3
                        idx_j = i - idx_i*3
                        residual.append(np.linalg.norm(
                            -rot_matrix_dis @ t_body_uwb[idx_j].T - transform_dis[0:3, 3] + t_body_uwb[idx_i].T) - distance)
                        residual2.append(residual[-1]**2)
                except ValueError:
                    # Skip malformed values
                    print(
                        f"Warning: Skipping malformed distance value: {uwb_data[distance_key]}")
                    continue
        print(f"Residuals_cal: {[f'{x:.4f}' for x in residual]}")
        print(np.sqrt(np.mean(residual2)))
        avg_uwb_distance = np.mean(uwb_distances) if uwb_distances else None

        return transform, transform_kf, avg_uwb_distance, transform_dis, uwb_dis_all

    except Exception as e:
        print(f"Error processing UWB data: {e}")
        # Return identity matrices as fallback
        return np.eye(4), np.eye(4), None, np.eye(4)


def read_submap_data(file_path):
    """Read submap data from CSV file"""
    try:
        submap_data = pd.read_csv(file_path)
        print(f"Loaded {len(submap_data)} submaps from {file_path}")
        return submap_data
    except Exception as e:
        print(f"Error reading submap file {file_path}: {e}")
        return None


def main():
    args = parse_args()
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Read submap data if provided
    submap_data_frames = []
    submap_id_to_timestamp = {}

    # Read submap data from each robot
    print("Reading submap data...")
    for i in range(args.num_robots):
        robot_name = ID2ROBOT[i]
        submap_csv_path = os.path.join(
            args.uwb_log_base_path, robot_name, 'distributed/kimera_distributed_submaps.csv')
        try:
            robot_submap_data = pd.read_csv(submap_csv_path)
            # Add robot ID to the submap data to track which robot each submap belongs to
            robot_submap_data['robot_id'] = i
            print(f"Loaded {len(robot_submap_data)} submaps from {robot_name}")
            # Add to frames list
            submap_data_frames.append(robot_submap_data)
            # Update lookup dictionary with composite key (robot_id, submap_id)
            for _, row in robot_submap_data.iterrows():
                # Create a composite key: (robot_id, submap_id)
                composite_key = (i, row['submap_id'])
                submap_id_to_timestamp[composite_key] = row['submap_stamp_ns']
        except Exception as e:
            print(f"Error reading submap file for {robot_name}: {e}")

    if submap_data_frames:
        submap_data = pd.concat(submap_data_frames, ignore_index=True)
        print(f"Total submaps combined: {len(submap_data)}")
        print(
            f"Created lookup table for {len(submap_id_to_timestamp)} submaps")
    else:
        submap_data = None
        print("No submap data could be loaded. Using UWB timestamps only.")

    # Read ground truth data for all robots
    print("Loading ground truth data...")
    groundtruth_files = [f"{args.gt_basic_path}/Kimera-Multi-Public-Data/ground_truth/{args.date}/modified_{ID2ROBOT[i]}_gt_odom.tum"
                         for i in range(args.num_robots)]
    groundtruth_data = {}
    for i, file in enumerate(tqdm(groundtruth_files, desc="Reading ground truth")):
        groundtruth_data[i] = read_groundtruth_tum(file)

    # Read UWB log data from multiple robots
    print(f"Reading UWB log data from {args.num_robots} robots...")
    uwb_data_frames = []
    for i in range(args.num_robots):
        robot_name = ID2ROBOT[i]
        uwb_log_path = os.path.join(
            args.uwb_log_base_path, robot_name, 'distributed/uwb_log.csv')

        try:
            robot_uwb_data = pd.read_csv(uwb_log_path)
            print(
                f"Loaded {len(robot_uwb_data)} UWB measurements from {robot_name}")
            uwb_data_frames.append(robot_uwb_data)
        except Exception as e:
            print(f"Error reading UWB log file for {robot_name}: {e}")

    # Concatenate all UWB data frames
    if not uwb_data_frames:
        print("No UWB data could be loaded. Exiting.")
        return

    uwb_data = pd.concat(uwb_data_frames, ignore_index=True)
    print(f"Total UWB measurements combined: {len(uwb_data)}")

    # Prepare output file
    output_csv = os.path.join(
        args.output_dir, f"uwb_evaluation_{args.date}.csv")
    with open(output_csv, 'w') as f:
        f.write("Index,Timestamp1,Timestamp2,Robot1,Robot2,"
                "SubMap1,SubMap2,"
                "GT_tx,GT_ty,GT_tz,GT_qx,GT_qy,GT_qz,GT_qw,"
                "UWB_tx,UWB_ty,UWB_tz,UWB_qx,UWB_qy,UWB_qz,UWB_qw,"
                "Translation_Error,Rotation_Error\n")

    # Create arrays to store results for plotting
    translation_errors = []
    rotation_errors = []
    distances = []

    # Process each UWB measurement
    print("Processing UWB measurements...")
    for idx, row in tqdm(uwb_data.iterrows(), total=len(uwb_data)):
        robot1 = int(row['robot1'])
        robot2 = int(row['robot2'])
        timestamp1 = None
        timestamp2 = None
        timestamp1_kf = float(row['stamp1'])
        timestamp2_kf = float(row['stamp2'])

        # If we have submap data, use it to find accurate timestamps
        if submap_data is not None and 'submap1' in row and 'submap2' in row:
            submap_id1 = row['submap1']
            submap_id2 = row['submap2']

            # Use composite keys (robot_id, submap_id) to look up timestamp
            composite_key1 = (robot1, submap_id1)
            composite_key2 = (robot2, submap_id2)

            if composite_key1 in submap_id_to_timestamp:
                # Use submap timestamp instead of UWB timestamp
                # Convert from ns to s
                timestamp1 = submap_id_to_timestamp[composite_key1] / 1e9
                print(
                    f"Using submap {submap_id1} from robot {robot1} timestamp: {timestamp1}")
                print(f"Related KF1's timestamp: {timestamp1_kf}")
            else:
                print(f"Could not find submap {submap_id1} for robot {robot1}")

            if composite_key2 in submap_id_to_timestamp:
                # Use submap timestamp instead of UWB timestamp
                # Convert from ns to s
                timestamp2 = submap_id_to_timestamp[composite_key2] / 1e9
                print(
                    f"Using submap {submap_id2} from robot {robot2} timestamp: {timestamp2}")
                print(f"Related KF2's timestamp: {timestamp2_kf}")
            else:
                print(f"Could not find submap {submap_id2} for robot {robot2}")

        if (timestamp1 is not None and timestamp2 is not None):
            # Find closest ground truth poses
            gt_pose1 = find_closest_pose(
                timestamp1, groundtruth_data[robot1], args.tolerance)
            gt_pose2 = find_closest_pose(
                timestamp2, groundtruth_data[robot2], args.tolerance)

            if gt_pose1 is not None and gt_pose2 is not None:
                # Convert ground truth poses to transformation matrices
                T_gt_1 = pose_to_transformation_matrix(gt_pose1)
                T_gt_2 = pose_to_transformation_matrix(gt_pose2)

                # Create transformation matrix from UWB data
                T_uwb, T_uwb_kf, avg_uwb_distance, T_uwb_dis, uwb_dis_all = uwb_transformation_matrix(
                    row)

                # Calculate error
                _, rot_error, trans_error = calculate_pose_error(
                    T_gt_1, T_gt_2, T_uwb)

                # Find closest ground truth poses (KF)
                gt_pose1_kf = find_closest_pose(
                    timestamp1_kf, groundtruth_data[robot1], args.tolerance)
                gt_pose2_kf = find_closest_pose(
                    timestamp2_kf, groundtruth_data[robot2], args.tolerance)
                T_gt_1_kf = pose_to_transformation_matrix(gt_pose1_kf)
                T_gt_2_kf = pose_to_transformation_matrix(gt_pose2_kf)
                _, rot_error_kf, trans_error_kf = calculate_pose_error(
                    T_gt_1_kf, T_gt_2_kf, T_uwb_kf)

                residual_gt = []
                residual_gt2 = []
                R_ij_kf = T_gt_1_kf[:3, :3].T @ T_gt_2_kf[:3, :3]
                t_ij_kf = T_gt_1_kf[:3,
                                    :3].T @ (T_gt_2_kf[:3, 3] - T_gt_1_kf[:3, 3])
                for i in range(9):
                    if (uwb_dis_all[i] > 0):
                        idx_i = i // 3
                        idx_j = i - idx_i * 3
                        diff_pos1 = - \
                            R_ij_kf @ t_body_uwb[idx_j].T - \
                            t_ij_kf + t_body_uwb[idx_i].T
                        # diff_pos2 = -T_gt_2_kf[:3, :3] @ t_body_uwb[idx_j].T - T_gt_2_kf[:3,
                        #                                                                  3] + T_gt_1_kf[:3, :3] @ t_body_uwb[idx_i].T + T_gt_1_kf[:3, 3]
                        # print(diff_pos1 - T_gt_1_kf[:3, :3].T @ diff_pos2)
                        residual_gt.append(
                            np.linalg.norm(diff_pos1, 2) - uwb_dis_all[i])
                        residual_gt2.append(residual_gt[-1]**2)
                        # residual_gt2.append(
                        #     np.linalg.norm(diff_pos2, 2) - uwb_dis_all[i])
                print(f"Residuals: {[f'{x:.4f}' for x in residual_gt]}")
                print(
                    f"Mean squared residual: {np.sqrt(np.mean(residual_gt2)):.4f}")

                # Find closest ground truth poses (dis)
                gt_pose1_dis = find_closest_pose(
                    timestamp1_kf, groundtruth_data[robot1], args.tolerance)
                gt_pose2_dis = find_closest_pose(
                    timestamp2_kf, groundtruth_data[robot2], args.tolerance)
                T_gt_1_dis = pose_to_transformation_matrix(gt_pose1_dis)
                T_gt_2_dis = pose_to_transformation_matrix(gt_pose2_dis)
                _, rot_error_dis, trans_error_dis = calculate_pose_error(
                    T_gt_1_dis, T_gt_2_dis, T_uwb_dis)

                rot_error_track, trans_error_track = None, None

                if T_gt_1_dis is not None and T_gt_2_dis is not None:
                    T_gt_12_dis = np.eye(4)
                    T_gt_12_dis[:3, :3] = T_gt_1_dis[:3,
                                                     : 3].T @ T_gt_2_dis[:3, :3]
                    T_gt_12_dis[:3, 3] = T_gt_1_dis[:3,
                                                    :3].T @ (T_gt_2_dis[:3, 3] - T_gt_1_dis[:3, 3])

                    _, rot_error_track, trans_error_track = calculate_pose_error(
                        T_gt_1_dis, T_gt_2_dis, T_gt_12_dis)

                # Calculate distance between robots from ground truth
                gt_distance = np.linalg.norm(gt_pose1_kf[:3] - gt_pose2_kf[:3])

                # Print distances and error for each measurement
                print(f"Measurement {idx} - Robot {robot1}→{robot2}")
                print(f"  GT distance: {gt_distance:.3f}m")
                if avg_uwb_distance is not None:
                    print(f"  Avg UWB distance: {avg_uwb_distance:.3f}m")
                    print(
                        f"  Distance error: {abs(gt_distance - avg_uwb_distance):.3f}m")
                print(
                    f"  Translation error: {trans_error:.4f}m, Rotation error: {rot_error:.4f}")
                print(
                    f"  KF - Translation error: {trans_error_kf:.4f}m, Rotation error: {rot_error_kf:.4f}")
                print(
                    f"  Distance-based - Translation error: {trans_error_dis:.4f}m, Rotation error: {rot_error_dis:.4f}")
                print(
                    f"  Est-kf-trans: {t_ij_kf.T}, Est_ij_trans: {T_uwb_dis[:3,3]}")
                print(
                    f"  True-ij-rot: {T_gt_12_dis[:3,:3]}, Est_ij_rot: {T_uwb_dis[:3,:3]}")
                # Store for plotting
                translation_errors.append(trans_error)
                rotation_errors.append(rot_error)
                distances.append(gt_distance)

                # Write to CSV
                with open(output_csv, 'a') as f:
                    f.write(f"{idx},{timestamp1},{timestamp2},{ID2ROBOT[robot1]},{ID2ROBOT[robot2]},"
                            f"{row['submap1']},{row['submap2']},"
                            f"{gt_pose1[0]},{gt_pose1[1]},{gt_pose1[2]},{gt_pose1[3]},{gt_pose1[4]},{gt_pose1[5]},{gt_pose1[6]},"
                            f"{row['tx']},{row['ty']},{row['tz']},{row['qx']},{row['qy']},{row['qz']},{row['qw']},"
                            f"{trans_error},{rot_error},{trans_error_kf},{rot_error_kf},{trans_error_dis},{rot_error_dis}\n")
            else:
                print(
                    f"Warning: Could not find matching GT poses for measurement {idx} with timestamp {timestamp1} and {timestamp2}")

    # Create plots
    print("Creating plots...")

    # Translation error vs distance
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, translation_errors, alpha=0.7)
    plt.xlabel('Distance between robots (m)')
    plt.ylabel('Translation error (m)')
    plt.title(f'UWB Translation Error vs Distance - Dataset {args.date}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.output_dir,
                f'translation_error_{args.date}.pdf'), dpi=300, bbox_inches='tight')

    # Rotation error vs distance
    plt.figure(figsize=(10, 6))
    plt.scatter(distances, rotation_errors, alpha=0.7)
    plt.xlabel('Distance between robots (m)')
    plt.ylabel('Rotation error (Frobenius norm)')
    plt.title(f'UWB Rotation Error vs Distance - Dataset {args.date}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(args.output_dir,
                f'rotation_error_{args.date}.pdf'), dpi=300, bbox_inches='tight')

    # Error histograms
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(translation_errors, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Translation error (m)')
    plt.ylabel('Count')
    plt.title('Translation Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.hist(rotation_errors, bins=30, alpha=0.7, color='red')
    plt.xlabel('Rotation error')
    plt.ylabel('Count')
    plt.title('Rotation Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir,
                f'error_histograms_{args.date}.pdf'), dpi=300, bbox_inches='tight')

    # Print statistics
    print(f"\nEvaluation completed. Results saved to {args.output_dir}")
    print(f"Total UWB measurements evaluated: {len(translation_errors)}")
    print(
        f"Translation error - Mean: {np.mean(translation_errors):.4f}m, Median: {np.median(translation_errors):.4f}m")
    print(
        f"Rotation error - Mean: {np.mean(rotation_errors):.4f}, Median: {np.median(rotation_errors):.4f}")


if __name__ == "__main__":
    main()
