'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2024-12-28 22:03:12
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-03-14 11:33:45
FilePath: /src/kimera_multi/evaluation/draw_lc.py
Version: 0.0.0
Description: Draw the loop closure result

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import os
import scienceplots

plt.style.use(['science', 'ieee', 'no-latex'])

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

#    blue,       orange,    green,     red,      purple,    brown,    pink,      gray,      yellow,     cyan
# ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#e377c2',
          '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', 'k']


def is_valid_number(value):
    """Check if value can be converted to float"""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def safe_eval_vector(vector_data):
    """Safely evaluate vector data from different formats"""
    try:
        if isinstance(vector_data, str):
            if 'No GT data' in vector_data:
                return None
            clean_str = vector_data.strip('[]')
            return np.array([float(x) for x in clean_str.split()])
        elif isinstance(vector_data, (list, np.ndarray)):
            return np.array(vector_data)
        elif isinstance(vector_data, (int, float)):
            return np.array([float(vector_data)])
        else:
            print(f"Unsupported vector data type: {type(vector_data)}")
            return None
    except Exception as e:
        print(f"Error parsing vector data '{vector_data}': {e}")
        return None


def read_groundtruth_tum(file_path):
    """Read the ground truth poses from TUM file"""
    groundtruth_data = pd.read_csv(file_path, sep=' ', header=None)
    groundtruth_data.columns = ['timestamp',
                                'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    return groundtruth_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Process loop closure results')
    parser.add_argument('--date', type=str, default='1207',
                        choices=['1207', '1014', '1208'],
                        help='Date of the dataset (e.g., 1207, 1014, 1208)')
    return parser.parse_args()


def process_lc_results(args, num_robots, groundtruth_prefix):

    # Setup visualization
    _, ax = plt.subplots(figsize=(10, 8))
    norm = Normalize(vmin=0, vmax=50)
    cmap = plt.cm.RdYlBu
    line_collection = []

    # Process each robot's data
    line_collection = []
    for i in range(num_robots):
        # Read ground truth trajectory
        gt_file = f"{groundtruth_prefix}modified_{ID2ROBOT[i]}_gt_odom.tum"
        gt_data = read_groundtruth_tum(gt_file)
        trajectory = gt_data[['tx', 'ty', 'tz']].values

        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1],
                label=ID2ROBOT[i], linewidth=1.5, color=COLORS[i])

        # Updated filename to match lc_result.py convention
        inter_filename = f"{args.date}/inter_lc_results_{args.date}_{ID2ROBOT[i]}.csv"
        intra_filename = f"{args.date}/intra_lc_results_{args.date}_{ID2ROBOT[i]}.csv"

        # Process inter-robot loop closures
        try:
            if os.path.exists(inter_filename):
                df = pd.read_csv(inter_filename)
                # Extract valid loop closure data
                valid_data = df.dropna(subset=['GT_Pose1_X', 'GT_Pose2_X',
                                               'GT_Pose1_Y', 'GT_Pose2_Y',
                                               'Distance'])

                # Plot loop closures
                for _, row in valid_data.iterrows():
                    try:
                        # Extract poses
                        pose1 = np.array(
                            [row['GT_Pose1_X'], row['GT_Pose1_Y']])
                        pose2 = np.array(
                            [row['GT_Pose2_X'], row['GT_Pose2_Y']])

                        # Get distance for coloring
                        distance = float(row['Distance'])
                        color = cmap(norm(distance))

                        # Draw loop closure connection
                        line = ax.plot([pose1[0], pose2[0]],
                                       [pose1[1], pose2[1]],
                                       '--',
                                       color=color,
                                       alpha=0.6,
                                       linewidth=1.0)
                        line_collection.append(line[0])

                        # Add distance value in the middle of the line
                        if distance >= 50:
                            mid_point = (pose1 + pose2) / 2
                            ax.text(mid_point[0], mid_point[1],
                                    f"{distance:.2f}",
                                    color=color,
                                    fontsize=8)

                        # Add markers for poses
                        ax.scatter(pose1[0], pose1[1],
                                   color=color,
                                   marker='o',
                                   s=20,
                                   zorder=5,
                                   edgecolor='black',
                                   linewidth=0.5)
                        ax.scatter(pose2[0], pose2[1],
                                   color=color,
                                   marker='s',
                                   s=20,
                                   zorder=5,
                                   edgecolor='black',
                                   linewidth=0.5)

                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue

        except Exception as e:
            print(f"Error reading file {inter_filename}: {e}")

        # Process intra-robot loop closures
        try:
            if os.path.exists(intra_filename):
                df = pd.read_csv(intra_filename)
                # Use different line style for intra-robot loop closures
                for _, row in df.dropna(subset=['GT_Pose1_X', 'GT_Pose2_X',
                                                'GT_Pose1_Y', 'GT_Pose2_Y',
                                                'Distance']).iterrows():
                    try:
                        # Extract poses
                        pose1 = np.array(
                            [row['GT_Pose1_X'], row['GT_Pose1_Y']])
                        pose2 = np.array(
                            [row['GT_Pose2_X'], row['GT_Pose2_Y']])

                        # Get distance for coloring
                        distance = float(row['Distance'])
                        # color = cmap(norm(distance))

                        # Add distance value in the middle of the line
                        if distance >= 30:
                            # Draw loop closure connection with different line style
                            line = ax.plot([pose1[0], pose2[0]],
                                           [pose1[1], pose2[1]],
                                           '-.',  # Different line style for intra-robot
                                           color=color,
                                           alpha=0.6,
                                           linewidth=1.0)
                            line_collection.append(line[0])

                            # TODO (Ronghai): temporarily disable distance text for LC
                            # mid_point = (pose1 + pose2) / 2
                            # ax.text(mid_point[0], mid_point[1],
                            #         f"{distance:.2f}",
                            #         color=color,
                            #         fontsize=8)

                        # Add markers for poses
                        ax.scatter(pose1[0], pose1[1],
                                   color=color,
                                   marker='o',
                                   s=20,
                                   zorder=5,
                                   edgecolor='black',
                                   linewidth=0.5)
                        ax.scatter(pose2[0], pose2[1],
                                   color=color,
                                   marker='o',
                                   s=20,
                                   zorder=5,
                                   edgecolor='black',
                                   linewidth=0.5)

                    except Exception as e:
                        print(f"Error processing row: {e}")
                        continue

        except Exception as e:
            print(f"Error reading file {intra_filename}: {e}")

    # ...existing visualization code...
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel(r'$X$ [m]', fontsize=12)
    ax.set_ylabel(r'$Y$ [m]', fontsize=12)
    # ax.set_title('Multi-Robot Trajectories and Loop Closures', fontsize=14)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_label('Loop Closure Distance (m)', fontsize=10)

    plt.legend()
    plt.savefig(f'processed_lc_results_{args.date}.pdf',
                dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    groundtruth_prefix = f"/media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/Kimera-Multi-Public-Data/ground_truth/{args.date}/"
    num_robots = 6
    # Remove input_prefix as it's now hardcoded
    process_lc_results(args, num_robots, groundtruth_prefix)
