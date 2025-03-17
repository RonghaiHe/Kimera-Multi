'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-12-30 12:19:40
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-03-14 11:48:25
FilePath: /src/kimera_multi/evaluation/extract_lc_images.py
Version: 1.0.0
Description: To extract images from rosbag based on Loop closure results (distance > 30m)

'''

#!/usr/bin/env python3

import argparse
import pandas as pd
import os
import subprocess
from tqdm import tqdm
import cv2
import time
import numpy as np
import multiprocessing
from functools import partial
from multiprocessing import Lock, Manager
import shutil
import uuid

ID2ROBOT = [
    'acl_jackal',
    'acl_jackal2',
    'sparkal1',
    'sparkal2',
    'hathor',
    'thoth',
    'apis',
    'sobek'
]

DATE2DATASET = {'1207': 'campus_tunnels_12_07',
                '1014': 'campus_outdoor_10_14',
                '1208': 'campus_hybrid_12_08'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract images from rosbag based on LC results')
    parser.add_argument('--date', type=str, default='1207',
                        choices=list(DATE2DATASET.keys()),
                        help='Date of the dataset (e.g., 1207, 1014, 1208)')
    parser.add_argument('--sta_basic_path', type=str,
                        default='/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets',
                        help='Base path to the estimation statistics of multi-robot datasets')
    parser.add_argument('--threshold', type=float, default=30.0,
                        help='Distance threshold for LC results')
    parser.add_argument('--basic_bag_path', type=str,
                        default='/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/',
                        help='Basic path to the rosbag file')
    return parser.parse_args()


def extract_and_merge_images(bag_file1, bag_file2, timestamp1, timestamp2, robot1, robot2, gt_pose1, gt_pose2, number, distance, score, mono_inliers, stereo_inliers, output_path, process_id, extraction_lock):
    """Extract images from two bags and merge them with process-specific temp directory"""

    # Create process-specific temporary directory
    process_temp_dir = os.path.dirname(
        output_path) + f'/temp_images_proc_{process_id}'
    os.makedirs(process_temp_dir, exist_ok=True)

    put_temp_dir = os.path.dirname(
        output_path) + f'/temp_images'

    # Create directory for original images
    orig_images_dir = os.path.join(
        os.path.dirname(output_path), 'original_images')
    os.makedirs(orig_images_dir, exist_ok=True)

    # Get base filename for consistent naming
    base_filename = os.path.basename(output_path).replace('.png', '')

    try:
        start_time1 = timestamp1 - 0.05
        end_time1 = timestamp1 + 0.05
        start_time2 = timestamp2 - 0.05
        end_time2 = timestamp2 + 0.05

        # Create process-specific temporary bags and image paths
        temp_bag1 = os.path.join(process_temp_dir, f'{number}_{robot1}_1.bag')
        temp_bag2 = os.path.join(process_temp_dir, f'{number}_{robot2}_2.bag')
        img1_path = os.path.join(process_temp_dir, f'{number}_1.jpg')
        img2_path = os.path.join(process_temp_dir, f'{number}_2.jpg')

        # Extract first image with lock
        if not os.path.exists(img1_path):
            subprocess.run(
                f'rosbag filter {bag_file1} {temp_bag1} "t.to_sec() >= {start_time1} and t.to_sec() <= {end_time1}"', shell=True)
            with extraction_lock:
                while True:
                    print(f"Extracting image 1 for {output_path}")
                    subprocess.run(
                        ['rosbag', 'play', temp_bag1],
                        check=True,
                        capture_output=True,
                        timeout=5  # 5 second timeout
                    )
                    time.sleep(5)

                    # Move the extracted frame file to the target directory if it exists,
                    # using a blur matching technique to ensure the correct frame is selected
                    if os.listdir(put_temp_dir):
                        shutil.move(
                            os.path.join(put_temp_dir,
                                         next(f for f in os.listdir(put_temp_dir)
                                              if f.startswith('frame'))),
                            img1_path)
                        break

        # Extract second image with lock
        if not os.path.exists(img2_path):
            subprocess.run(
                f'rosbag filter {bag_file2} {temp_bag2} "t.to_sec() >= {start_time2} and t.to_sec() <= {end_time2}"', shell=True)
            with extraction_lock:
                while True:
                    print(f"Extracting image 2 for {output_path}")
                    subprocess.run(
                        ['rosbag', 'play', temp_bag2],
                        check=True,
                        capture_output=True,
                        timeout=5  # 5 second timeout
                    )
                    time.sleep(5)

                    # Move the extracted frame file to the target directory if it exists,
                    # using a blur matching technique to ensure the correct frame is selected
                    if os.listdir(put_temp_dir):
                        shutil.move(
                            os.path.join(put_temp_dir,
                                         next(f for f in os.listdir(put_temp_dir)
                                              if f.startswith('frame'))),
                            img2_path)
                        break

        # Read and process images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise ValueError(f"Failed to read images for {output_path}")

        # Save original images to original_images directory
        orig_img1_path = os.path.join(
            orig_images_dir, f'{base_filename}_img1.jpg')
        orig_img2_path = os.path.join(
            orig_images_dir, f'{base_filename}_img2.jpg')

        # After reading the images, save copies of the originals
        if img1 is not None and img2 is not None:
            cv2.imwrite(orig_img1_path, img1)
            cv2.imwrite(orig_img2_path, img2)

        # Resize images to same height
        height = min(img1.shape[0], img2.shape[0])
        width = min(img1.shape[1], img2.shape[1])
        img1 = cv2.resize(img1, (width, height))
        img2 = cv2.resize(img2, (width, height))

        # Merge images horizontally
        merged_img = np.hstack((img1, img2))

        # Get dimensions of merged image
        height, width = merged_img.shape[:2]

        # Add padding (top, bottom, left, right)
        # Increased top and bottom padding for more text
        padding = ((80, 50), (10, 10), (0, 0))  # Increased bottom padding
        merged_img = np.pad(merged_img, padding, mode='constant')

        # Calculate text positions
        text_color = (255, 255, 255)  # White
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2

        # Distance
        cv2.putText(merged_img,
                    f'Dist: {distance:.2f}',
                    (20, 30),
                    font, font_scale, text_color, thickness)

        # Loop closure score
        cv2.putText(merged_img,
                    f'Score: {score}',
                    (240, 30),
                    font, font_scale, text_color, thickness)

        # Mono and stereo inliers
        cv2.putText(merged_img,
                    f'Mono: {mono_inliers}',
                    (460, 30),
                    font, font_scale, text_color, thickness)
        cv2.putText(merged_img,
                    f'Stereo: {stereo_inliers}',
                    (680, 30),
                    font, font_scale, text_color, thickness)

        # Groundtruth poses
        cv2.putText(merged_img,
                    gt_pose1,
                    (20, 60),
                    font, font_scale, text_color, thickness)
        cv2.putText(merged_img,
                    gt_pose2,
                    (width//2 + 20, 60),
                    font, font_scale, text_color, thickness)

        # Add text with adjusted positions
        # Robot names and timestamps
        cv2.putText(merged_img,
                    f'{robot1} {timestamp1:.2f}',
                    (20, height + 100),
                    font, font_scale, text_color, thickness)
        cv2.putText(merged_img,
                    f'{robot2} {timestamp2:.2f}',
                    (width//2 + 20, height + 100),
                    font, font_scale, text_color, thickness)

        # Write final image with lock
        with extraction_lock:
            cv2.imwrite(output_path, merged_img)

        return True

    except Exception as e:
        print(f"Error processing images for {output_path}: {str(e)}")
        return False

    finally:
        # Cleanup process-specific temporary files
        if os.path.exists(process_temp_dir):
            shutil.rmtree(process_temp_dir)


def process_loop_closure(row, bag_paths, output_dir, extraction_lock):
    """Process a single loop closure with process-specific ID and lock"""
    process_id = str(uuid.uuid4())[:8]  # Generate unique process ID

    timestamp1 = row['Timestamp 1']
    timestamp2 = row['Timestamp 2']
    number = row['Loop Closure Number']
    distance = row['Distance']
    robot1 = row['Robot 1']
    robot2 = row['Robot 2']

    gt_pose1 = f"GT Pose1: {row['GT_Pose1_X']:.2f}, {row['GT_Pose1_Y']:.2f}, {row['GT_Pose1_Z']:.2f}"
    gt_pose2 = f"GT Pose2: {row['GT_Pose2_X']:.2f}, {row['GT_Pose2_Y']:.2f}, {row['GT_Pose2_Z']:.2f}"

    score = '-'
    # if row exists 'norm_bow_score'
    if 'norm_bow_score' in row:
        score = f"{row['norm_bow_score']:.2f}"

    mono_inliers = row['mono_inliers']
    stereo_inliers = row['stereo_inliers']

    if timestamp1 > 1e12:
        timestamp1 = timestamp1 / 1e9

    if timestamp2 > 1e12:
        timestamp2 = timestamp2 / 1e9

    output_image_path = os.path.join(output_dir, 'lc_')
    output_image_path += 'intra_' if robot1 == robot2 else 'inter_'
    output_image_path += f'{distance:.2f}_{robot1}_{robot2}_{number}.png'

    # Check if file exists with lock
    with extraction_lock:
        if os.path.exists(output_image_path):
            return output_image_path

    success = extract_and_merge_images(
        bag_paths[robot1],
        bag_paths[robot2],
        timestamp1,
        timestamp2,
        robot1,
        robot2,
        gt_pose1,
        gt_pose2,
        number,
        distance,
        score,
        mono_inliers,
        stereo_inliers,
        output_image_path,
        process_id,
        extraction_lock
    )

    return output_image_path if success else None


def main():
    args = parse_args()

    bag_paths = {
        robot_name: f'{args.basic_bag_path}{DATE2DATASET[args.date]}/{args.date[:2]}_{args.date[2:]}_{robot_name}.bag'
        for robot_name in ID2ROBOT
    }

    # Read the LC results CSV files
    all_high_distance_rows = []
    print("Reading CSV files...")
    for robot_name in tqdm(ID2ROBOT, desc="Processing robots"):
        inter_csv_filename = args.sta_basic_path + \
            f'/{args.date}/inter_lc_results_{args.date}_{robot_name}.csv'
        if os.path.exists(inter_csv_filename):
            try:
                df = pd.read_csv(inter_csv_filename)
            except Exception as e:
                print(f"Error reading CSV file {inter_csv_filename}: {str(e)}")
                raise
            # Convert Distance column to float and handle any non-numeric values
            df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
            high_distance_rows = df[df['Distance'] > args.threshold]
            all_high_distance_rows.extend(
                high_distance_rows.to_dict('records'))

        intra_csv_filename = args.sta_basic_path + \
            f'/{args.date}/intra_lc_results_{args.date}_{robot_name}.csv'
        if os.path.exists(intra_csv_filename):
            df = pd.read_csv(intra_csv_filename)
            # Convert Distance column to float and handle any non-numeric values
            df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
            high_distance_rows = df[df['Distance'] > args.threshold]
            if not high_distance_rows.empty:
                high_distance_rows.loc[:, 'Robot 1'] = robot_name
                high_distance_rows.loc[:, 'Robot 2'] = robot_name
            all_high_distance_rows.extend(
                high_distance_rows.to_dict('records'))

    if not all_high_distance_rows:
        print(f"No distances found above threshold {args.threshold}")
        return

    # Create output directory
    output_dir = args.sta_basic_path + \
        f'/lc_images_{args.date}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '/temp_images', exist_ok=True)
    # Create directory for original images
    os.makedirs(output_dir + '/original_images', exist_ok=True)

    # Create a manager for sharing the lock between processes
    manager = Manager()
    extraction_lock = manager.Lock()

    # Process loop closures in parallel
    print("\nProcessing loop closures...")
    num_processes = 8  # max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(num_processes) as pool:
        process_func = partial(process_loop_closure,
                               bag_paths=bag_paths,
                               output_dir=output_dir,
                               extraction_lock=extraction_lock)

        results = list(tqdm(
            pool.imap(process_func, all_high_distance_rows),
            total=len(all_high_distance_rows),
            desc=f"Extracting images using {num_processes} processes"
        ))

    # Count successful extractions
    successful_results = [r for r in results if r is not None]
    print(
        f'Successfully processed {len(successful_results)} out of {len(results)} loop closures')


if __name__ == '__main__':
    main()
