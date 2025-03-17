'''
Copyright © 2025, Sun Yat-sen University, Guangzhou, Guangdong, 510275, All Rights Reserved
Author: Ronghai He
Date: 2025-01-07 18:07:54
LastEditors: RonghaiHe hrhkjys@qq.com
LastEditTime: 2025-03-15 10:14:37
FilePath: /src/kimera_multi/evaluation/analyze_inliers.py
Version: 1.0.0
Description: To analyze monocular inliers and stereo inliers from loop closure

'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import argparse
import os
import numpy as np
from math import acos, sqrt, degrees

'''
python analyze_inliers.py --directory /media/sysu/Data/multi_robot_datasets/kimera_multi_datasets/evo_try/campus_tunnels_12_07/test_distributed_mono_count_kf_undistort/test_distributed3/1207/
'''


def calculate_translation_error(x, y, z):
    """Calculate translation error magnitude"""
    try:
        x, y, z = float(x), float(y), float(z)
        return sqrt(x**2 + y**2 + z**2)
    except (ValueError, TypeError):
        return float('inf')


def calculate_rotation_error_degrees(qw, qx, qy, qz):
    """Calculate rotation error in degrees from quaternion

    This implements the proper quaternion-to-angle conversion that works for all rotation magnitudes.
    For a rotation quaternion q = [qw, qx, qy, qz], the rotation angle in radians is:
    2 * arccos(|qw|)
    """
    try:
        qw, qx, qy, qz = float(qw), float(qx), float(qy), float(qz)

        # Normalize the quaternion first to ensure numerical stability
        norm = sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        if norm < 1e-10:  # Avoid division by extremely small numbers
            return float('inf')

        qw /= norm

        # The rotation angle is 2*arccos(|qw|)
        # We use abs(qw) because q and -q represent the same rotation
        angle_rad = 2 * acos(min(1.0, abs(qw)))
        angle_deg = degrees(angle_rad)

        return angle_deg
    except (ValueError, TypeError):
        return float('inf')


def is_good_transformation(trans_error, rot_error, trans_threshold=1.0, rot_threshold=10.0):
    """Check if transformation errors are within thresholds"""
    return trans_error <= trans_threshold and rot_error <= rot_threshold


def categorize_transformation(trans_error, rot_error):
    """Categorize the transformation based on translation and rotation errors"""
    if trans_error <= 1.0 and rot_error <= 10.0:
        return 'Good (≤1.0m, ≤10°)'
    elif trans_error <= 2.0 and rot_error <= 15.0:
        return 'Fair (≤2.0m, ≤15°)'
    elif trans_error <= 5.0 and rot_error <= 30.0:
        return 'Poor (≤5.0m, ≤30°)'
    else:
        return 'Bad (>5.0m, >30°)'


def extract_translation_from_vector(vector_str):
    """Extract x, y, z components from a translation vector string like '[1.2 -3.4 5.6]'"""
    try:
        # Remove brackets and split by spaces
        values = vector_str.strip('[]').split()

        # Convert to float and return x, y, z
        if len(values) >= 3:
            return float(values[0]), float(values[1]), float(values[2])
        return float('inf'), float('inf'), float('inf')
    except (ValueError, AttributeError, TypeError):
        return float('inf'), float('inf'), float('inf')


def extract_quaternion_from_vector(vector_str):
    """Extract w, x, y, z components from a rotation quaternion string like '[0.1 0.2 0.3 0.4]'"""
    try:
        # Remove brackets and split by spaces
        values = vector_str.strip('[]').split()

        # For quaternion format [x y z w]
        if len(values) >= 4:
            # Reorder if needed - some formats store as [x y z w] instead of [w x y z]
            # Detect format by checking which value has largest magnitude (usually w)
            vals = [float(v) for v in values]
            if abs(vals[3]) > max(abs(vals[0]), abs(vals[1]), abs(vals[2])):
                # Format is [x y z w]
                return float(vals[3]), float(vals[0]), float(vals[1]), float(vals[2])
            elif abs(vals[0]) > max(abs(vals[1]), abs(vals[2]), abs(vals[3])):
                # Format is [w x y z]
                return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

        # Default to identity quaternion if format can't be determined
        return 1.0, 0.0, 0.0, 0.0
    except (ValueError, AttributeError, TypeError):
        return 1.0, 0.0, 0.0, 0.0


def load_and_process_csv(directory):
    # Convert to absolute path if relative
    abs_directory = os.path.abspath(directory)
    file_pattern = os.path.join(abs_directory, "*_lc_results_*.csv")
    all_data = []

    print(f"Looking for CSV files in: {abs_directory}")
    csv_files = glob.glob(file_pattern)
    print(f"Found {len(csv_files)} CSV files")

    for file in csv_files:
        print(f"Processing file: {os.path.basename(file)}")
        try:
            df = pd.read_csv(file)
            # Filter out "No GT data" rows if they exist
            if 'Distance' in df.columns:
                df = df[df['Distance'] != "No GT data"]
            df['type'] = 'inter' if 'inter' in file else 'intra'

            # For files missing required columns, try to extract them from other columns
            missing_translation = not all(col in df.columns for col in [
                                          'trans_x', 'trans_y', 'trans_z'])
            missing_quaternion = not all(col in df.columns for col in [
                                         'quat_w', 'quat_x', 'quat_y', 'quat_z'])

            if missing_translation or missing_quaternion:
                # Check if we have relative transformation vectors
                if 'Relative Translation Vector' in df.columns:
                    print(
                        f"Extracting translation from 'Relative Translation Vector' in {os.path.basename(file)}")
                    # Extract translation components
                    df[['trans_x', 'trans_y', 'trans_z']] = df['Relative Translation Vector'].apply(
                        lambda x: pd.Series(extract_translation_from_vector(x))
                    )
                    missing_translation = False

                if 'Relative Rotation Quaternion' in df.columns:
                    print(
                        f"Extracting rotation from 'Relative Rotation Quaternion' in {os.path.basename(file)}")
                    # Extract quaternion components
                    df[['quat_w', 'quat_x', 'quat_y', 'quat_z']] = df['Relative Rotation Quaternion'].apply(
                        lambda x: pd.Series(extract_quaternion_from_vector(x))
                    )
                    missing_quaternion = False

                # As fallback, check for estimated values
                if missing_translation and 'Estimated Relative Translation' in df.columns:
                    print(
                        f"Using 'Estimated Relative Translation' as fallback in {os.path.basename(file)}")
                    df[['trans_x', 'trans_y', 'trans_z']] = df['Estimated Relative Translation'].apply(
                        lambda x: pd.Series(extract_translation_from_vector(x))
                    )
                    missing_translation = False

                if missing_quaternion and 'Estimated Relative Rotation' in df.columns:
                    print(
                        f"Using 'Estimated Relative Rotation' as fallback in {os.path.basename(file)}")
                    df[['quat_w', 'quat_x', 'quat_y', 'quat_z']] = df['Estimated Relative Rotation'].apply(
                        lambda x: pd.Series(extract_quaternion_from_vector(x))
                    )
                    missing_quaternion = False

            # If still missing translation after attempts to extract
            if not all(col in df.columns for col in ['trans_x', 'trans_y', 'trans_z']):
                print(
                    f"Warning: File {os.path.basename(file)} missing translation columns")
                df['translation_error'] = float('inf')
                # Add missing columns to avoid future errors
                for col in ['trans_x', 'trans_y', 'trans_z']:
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                df['translation_error'] = df.apply(
                    lambda row: calculate_translation_error(
                        row['trans_x'], row['trans_y'], row['trans_z']), axis=1)

            if not all(col in df.columns for col in ['quat_w', 'quat_x', 'quat_y', 'quat_z']):
                print(
                    f"Warning: File {os.path.basename(file)} missing quaternion columns")
                df['rotation_error_deg'] = float('inf')
                # Add missing columns to avoid future errors
                for col in ['quat_w', 'quat_x', 'quat_y', 'quat_z']:
                    if col not in df.columns:
                        df[col] = np.nan
            else:
                df['rotation_error_deg'] = df.apply(
                    lambda row: calculate_rotation_error_degrees(
                        row['quat_w'], row['quat_x'], row['quat_y'], row['quat_z']), axis=1)

            # Ensure we have mono_inliers and stereo_inliers columns
            if 'mono_inliers' not in df.columns:
                df['mono_inliers'] = 0
            if 'stereo_inliers' not in df.columns:
                df['stereo_inliers'] = 0

            # Add transformation quality indicators
            df['good_transformation'] = df.apply(
                lambda row: is_good_transformation(
                    row['translation_error'], row['rotation_error_deg']), axis=1)
            df['transformation_category'] = df.apply(
                lambda row: categorize_transformation(
                    row['translation_error'], row['rotation_error_deg']), axis=1)

            # Print some debugging information
            print(f"  Rows: {len(df)}")
            print(f"  Good transformations: {df['good_transformation'].sum()}")

            # Validate the data quality
            if df['translation_error'].min() == float('inf') or df['rotation_error_deg'].min() == float('inf'):
                print(
                    f"  Warning: File {os.path.basename(file)} has invalid transformation data")
            else:
                print(
                    f"  Translation error min: {df['translation_error'].min():.3f}, max: {df['translation_error'].max():.3f}")
                print(
                    f"  Rotation error min: {df['rotation_error_deg'].min():.3f}°, max: {df['rotation_error_deg'].max():.3f}°")

            all_data.append(df)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue with next file instead of failing completely
            continue

    if not all_data:
        raise ValueError(
            f"No valid CSV files found matching pattern: {file_pattern}")

    # Combine all dataframes, handling different column sets
    try:
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)

        # Ensure critical columns exist after concatenation
        for col in ['translation_error', 'rotation_error_deg', 'good_transformation', 'transformation_category']:
            if col not in combined_df.columns:
                print(f"Warning: Missing column '{col}' after concatenation")
                if col == 'good_transformation':
                    combined_df[col] = False  # Default to False
                elif col == 'transformation_category':
                    combined_df[col] = 'Bad (>5.0m, >30°)'  # Default category
                else:
                    combined_df[col] = float('inf')  # Default infinite error
    except Exception as e:
        print(f"Error during dataframe concatenation: {str(e)}")
        # Create a minimal dataframe for analysis to continue
        combined_df = pd.DataFrame({
            'type': [],
            'mono_inliers': [],
            'stereo_inliers': [],
            'translation_error': [],
            'rotation_error_deg': [],
            'good_transformation': [],
            'transformation_category': []
        })

    # Final verification that columns exist
    print("\nCombined dataframe information:")
    print(f"Total rows: {len(combined_df)}")
    print(f"Columns: {', '.join(combined_df.columns)}")
    if 'good_transformation' in combined_df.columns:
        print(
            f"Good transformations: {combined_df['good_transformation'].sum()}")
    else:
        print("Warning: 'good_transformation' column is missing")

    return combined_df


def categorize_distance(distance):
    try:
        # Convert string to float for comparison
        dist = float(distance)
        if dist <= 10:
            return '0-10m'
        elif dist <= 30:
            return '10-30m'
        else:
            return '>30m'
    except (ValueError, TypeError):
        return 'invalid'


def categorize_mono_inliers(inliers):
    try:
        val = int(inliers)
        if val == 10:
            return '10'
        elif val == 11:
            return '11'
        elif val == 12:
            return '12'
        elif val == 13:
            return '13'
        elif val == 14:
            return '14'
        elif val == 15:
            return '15'
        elif val > 15:
            return '>15'
        else:
            return '<10'
    except (ValueError, TypeError):
        return 'invalid'


def categorize_stereo_inliers(inliers):
    try:
        val = int(inliers)
        if val == 5:
            return '5'
        elif val == 6:
            return '6'
        elif val == 7:
            return '7'
        elif val == 8:
            return '8'
        elif val == 9:
            return '9'
        elif val == 10:
            return '10'
        elif val > 10:
            return '>10'
        else:
            return '<5'
    except (ValueError, TypeError):
        return 'invalid'


def format_value(x):
    """Format value to show 0 instead of empty or invalid"""
    return str(int(x)) if x is not None else '0'
    # try:
    #     val = int(x)
    #     if val == 0:
    #         return '0'  # Explicitly return '0' for zero values
    #     return str(val)
    # except (ValueError, TypeError):
    #     return '0'


def analyze_inliers(directory):
    try:
        # Load data
        data = load_and_process_csv(directory)

        # Check if we have data to analyze
        if len(data) == 0:
            print("No data to analyze. Exiting...")
            return

        print(f"\nAnalysis Results:")
        print(f"Total data points: {len(data)}")

        # Filter data for good transformations or handle missing column
        if 'good_transformation' in data.columns:
            # Ensure column is boolean type
            if data['good_transformation'].dtype != bool:
                data['good_transformation'] = data['good_transformation'].astype(
                    bool)
            good_data = data[data['good_transformation'] == True]
            print(
                f"Good transformations (trans ≤ 1.0m, rot ≤ 10°): {len(good_data)}")
        else:
            print("Warning: No transformation quality data available")
            # Use all data as fallback
            good_data = data
            data['good_transformation'] = False  # Add column for later use

        # If no good transformations, use distance-based approach instead
        if len(good_data) == 0:
            print("No good transformations found. Using distance-based analysis instead.")
            # Categorize by distance instead of transformation quality
            data['distance_category'] = pd.Categorical(
                data['Distance'].apply(categorize_distance) if 'Distance' in data.columns
                else 'unknown',
                categories=['0-10m', '10-30m', '>30m', 'unknown'],
                ordered=True
            )

            # Original distance-based analysis
            analyze_by_distance(data)
            return

        # Continue with transformation-based analysis if we have good data
        # Add categorizations based on transformation quality instead of distance
        good_data['transformation_category'] = pd.Categorical(
            good_data['transformation_category'],
            categories=['Good (≤1.0m, ≤10°)', 'Fair (≤2.0m, ≤15°)',
                        'Poor (≤5.0m, ≤30°)', 'Bad (>5.0m, >30°)'],
            ordered=True
        )
        good_data['mono_category'] = pd.Categorical(
            good_data['mono_inliers'].apply(categorize_mono_inliers),
            categories=['>15', '15', '14', '13', '12', '11', '10', '<10'],
            ordered=True
        )
        good_data['stereo_category'] = pd.Categorical(
            good_data['stereo_inliers'].apply(categorize_stereo_inliers),
            categories=['>10', '10', '9', '8', '7', '6', '5', '<5'],
            ordered=True
        )

        # Split by type and create pivot tables
        inter_data = good_data[good_data['type'] == 'inter']
        intra_data = good_data[good_data['type'] == 'intra']

        # Create pivot tables with zeros for missing combinations
        processed_data = {}
        for type_name, type_data in [('inter', inter_data), ('intra', intra_data)]:
            # Create all possible combinations
            # Only good transformations for visualization
            transform_cats = ['Good (≤1.0m, ≤10°)']
            mono_cats = ['>15', '15', '14', '13', '12', '11', '10', '<10']
            stereo_cats = ['>10', '10', '9', '8', '7', '6', '5', '<5']

            # Create complete index for mono
            mono_idx = pd.MultiIndex.from_product([transform_cats, mono_cats],
                                                  names=['transformation_category', 'mono_category'])
            mono_counts = pd.Series(0, index=mono_idx).reset_index()

            # Create complete index for stereo
            stereo_idx = pd.MultiIndex.from_product([transform_cats, stereo_cats],
                                                    names=['transformation_category', 'stereo_category'])
            stereo_counts = pd.Series(0, index=stereo_idx).reset_index()

            # Count actual occurrences
            actual_mono = type_data.groupby(
                ['transformation_category', 'mono_category'], observed=True).size().reset_index(name='count')
            actual_stereo = type_data.groupby(
                ['transformation_category', 'stereo_category'], observed=True).size().reset_index(name='count')

            # Merge with complete combinations
            mono_final = mono_counts.merge(actual_mono, how='left',
                                           on=['transformation_category', 'mono_category']).fillna(0)
            stereo_final = stereo_counts.merge(actual_stereo, how='left',
                                               on=['transformation_category', 'stereo_category']).fillna(0)

            processed_data[type_name] = {
                'mono': mono_final,
                'stereo': stereo_final
            }

        # Create 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

        # Plot Inter Mono Inliers
        sns.barplot(data=processed_data['inter']['mono'],
                    x='transformation_category',
                    y='count',
                    hue='mono_category',
                    ax=ax1)
        ax1.set_title('Inter-frame Mono Inliers Distribution')
        ax1.set_ylabel('Count')
        ax1.set_xlabel('Transformation Quality')
        for container in ax1.containers:
            ax1.bar_label(container, fmt='%d', padding=3)

        # Plot Intra Mono Inliers
        sns.barplot(data=processed_data['intra']['mono'],
                    x='transformation_category',
                    y='count',
                    hue='mono_category',
                    ax=ax2)
        ax2.set_title('Intra-frame Mono Inliers Distribution')
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Transformation Quality')
        for container in ax2.containers:
            ax2.bar_label(container, fmt='%d', padding=3)

        # Plot Inter Stereo Inliers
        sns.barplot(data=processed_data['inter']['stereo'],
                    x='transformation_category',
                    y='count',
                    hue='stereo_category',
                    ax=ax3)
        ax3.set_title('Inter-frame Stereo Inliers Distribution')
        ax3.set_ylabel('Count')
        ax3.set_xlabel('Transformation Quality')
        for container in ax3.containers:
            ax3.bar_label(container, fmt='%d', padding=3)

        # Plot Intra Stereo Inliers
        sns.barplot(data=processed_data['intra']['stereo'],
                    x='transformation_category',
                    y='count',
                    hue='stereo_category',
                    ax=ax4)
        ax4.set_title('Intra-frame Stereo Inliers Distribution')
        ax4.set_ylabel('Count')
        ax4.set_xlabel('Transformation Quality')
        for container in ax4.containers:
            ax4.bar_label(container, fmt='%d', padding=3)

        plt.tight_layout()
        plt.savefig('inliers_analysis.jpg')

        # print the data that if mono_inliners < 10
        filtered_data = good_data[good_data['mono_inliers'] < 10]
        with open('mono_less_10.csv', 'w') as f:
            filtered_data.to_csv(f)

        # Create additional visualizations to analyze transformation errors and inliers
        create_transformation_inlier_plots(data)

        # Calculate and save transformation category ratios instead of distance
        output_transformation_ratios(data)

        # Save the transformation error statistics
        output_transformation_stats(data)

    except Exception as e:
        print(f"Error in analyze_inliers: {str(e)}")
        import traceback
        traceback.print_exc()


def analyze_by_distance(data):
    """Fall back to the original distance-based analysis when no transformation data is available"""

    # Add categorizations
    if 'Distance' in data.columns:
        data['distance_category'] = pd.Categorical(
            data['Distance'].apply(categorize_distance),
            categories=['0-10m', '10-30m', '>30m', 'unknown'],
            ordered=True
        )
    else:
        data['distance_category'] = pd.Categorical(
            ['unknown'] * len(data),
            categories=['0-10m', '10-30m', '>30m', 'unknown'],
            ordered=True
        )

    data['mono_category'] = pd.Categorical(
        data['mono_inliers'].apply(categorize_mono_inliers),
        categories=['>15', '15', '14', '13',
                    '12', '11', '10', '<10', 'invalid'],
        ordered=True
    )
    data['stereo_category'] = pd.Categorical(
        data['stereo_inliers'].apply(categorize_stereo_inliers),
        categories=['>10', '10', '9', '8', '7', '6', '5', '<5', 'invalid'],
        ordered=True
    )

    # Split by type and create pivot tables
    inter_data = data[data['type'] == 'inter']
    intra_data = data[data['type'] == 'intra']

    # Create pivot tables with zeros for missing combinations
    processed_data = {}
    for type_name, type_data in [('inter', inter_data), ('intra', intra_data)]:
        # Create all possible combinations
        dist_cats = ['0-10m', '10-30m', '>30m', 'unknown']
        mono_cats = ['>15', '15', '14', '13', '12', '11', '10', '<10']
        stereo_cats = ['>10', '10', '9', '8', '7', '6', '5', '<5']

        # Create complete index for mono
        mono_idx = pd.MultiIndex.from_product([dist_cats, mono_cats],
                                              names=['distance_category', 'mono_category'])
        mono_counts = pd.Series(0, index=mono_idx).reset_index()

        # Create complete index for stereo
        stereo_idx = pd.MultiIndex.from_product([dist_cats, stereo_cats],
                                                names=['distance_category', 'stereo_category'])
        stereo_counts = pd.Series(0, index=stereo_idx).reset_index()

        # Count actual occurrences
        actual_mono = type_data.groupby(
            ['distance_category', 'mono_category'], observed=True).size().reset_index(name='count')
        actual_stereo = type_data.groupby(
            ['distance_category', 'stereo_category'], observed=True).size().reset_index(name='count')

        # Merge with complete combinations
        mono_final = mono_counts.merge(actual_mono, how='left',
                                       on=['distance_category', 'mono_category']).fillna(0)
        stereo_final = stereo_counts.merge(actual_stereo, how='left',
                                           on=['distance_category', 'stereo_category']).fillna(0)

        processed_data[type_name] = {
            'mono': mono_final,
            'stereo': stereo_final
        }

    # Create 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))

    # Plot Inter Mono Inliers
    sns.barplot(data=processed_data['inter']['mono'],
                x='distance_category',
                y='count',
                hue='mono_category',
                ax=ax1)
    ax1.set_title('Inter-frame Mono Inliers Distribution')
    ax1.set_ylabel('Count')
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%d', padding=3)

    # Plot Intra Mono Inliers
    sns.barplot(data=processed_data['intra']['mono'],
                x='distance_category',
                y='count',
                hue='mono_category',
                ax=ax2)
    ax2.set_title('Intra-frame Mono Inliers Distribution')
    ax2.set_ylabel('Count')
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%d', padding=3)

    # Plot Inter Stereo Inliers
    sns.barplot(data=processed_data['inter']['stereo'],
                x='distance_category',
                y='count',
                hue='stereo_category',
                ax=ax3)
    ax3.set_title('Inter-frame Stereo Inliers Distribution')
    ax3.set_ylabel('Count')
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%d', padding=3)

    # Plot Intra Stereo Inliers
    sns.barplot(data=processed_data['intra']['stereo'],
                x='distance_category',
                y='count',
                hue='stereo_category',
                ax=ax4)
    ax4.set_title('Intra-frame Stereo Inliers Distribution')
    ax4.set_ylabel('Count')
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%d', padding=3)

    plt.tight_layout()
    plt.savefig('inliers_analysis.jpg')

    # Print the data that if mono_inliners < 10
    filtered_data = data[data['mono_inliers'] < 10]
    with open('mono_less_10.csv', 'w') as f:
        filtered_data.to_csv(f)

    # Output distance category ratios
    output_distance_ratios(data)


def output_transformation_ratios(data):
    """Calculate and output the ratio of transformation categories to a file"""
    # Get total count
    total_count = len(data)

    # Count by transformation category
    transform_counts = data['transformation_category'].value_counts(
    ).sort_index()

    # Calculate ratios
    transform_ratios = transform_counts / total_count * 100

    # Create a DataFrame for the results
    result_df = pd.DataFrame({
        'Category': transform_counts.index,
        'Count': transform_counts.values,
        'Ratio (%)': transform_ratios.values.round(2)
    })

    # Save to file
    result_df.to_csv('transformation_category_ratios.csv', index=False)

    # Also create a text file with more readable format
    with open('transformation_category_ratios.txt', 'w') as f:
        f.write("Transformation Category Distribution\n")
        f.write("----------------------------------\n")
        f.write(f"Total frames analyzed: {total_count}\n\n")

        for idx, row in result_df.iterrows():
            f.write(
                f"{row['Category']}: {int(row['Count'])} frames ({row['Ratio (%)']:.2f}%)\n")


def create_transformation_inlier_plots(data):
    """Create additional visualizations relating transformation errors to inliers"""
    plt.figure(figsize=(10, 8))
    plt.scatter(data['translation_error'], data['mono_inliers'], alpha=0.5)
    plt.axvline(x=1.0, color='r', linestyle='--', label='Threshold (1.0m)')
    plt.xlabel('Translation Error (m)')
    plt.ylabel('Monocular Inliers')
    plt.title('Relationship Between Translation Error and Mono Inliers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('translation_vs_mono_inliers.jpg')

    plt.figure(figsize=(10, 8))
    plt.scatter(data['rotation_error_deg'], data['mono_inliers'], alpha=0.5)
    plt.axvline(x=10.0, color='r', linestyle='--', label='Threshold (10°)')
    plt.xlabel('Rotation Error (degrees)')
    plt.ylabel('Monocular Inliers')
    plt.title('Relationship Between Rotation Error and Mono Inliers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rotation_vs_mono_inliers.jpg')

    plt.figure(figsize=(10, 8))
    plt.scatter(data['translation_error'], data['stereo_inliers'], alpha=0.5)
    plt.axvline(x=1.0, color='r', linestyle='--', label='Threshold (1.0m)')
    plt.xlabel('Translation Error (m)')
    plt.ylabel('Stereo Inliers')
    plt.title('Relationship Between Translation Error and Stereo Inliers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('translation_vs_stereo_inliers.jpg')

    plt.figure(figsize=(10, 8))
    plt.scatter(data['rotation_error_deg'], data['stereo_inliers'], alpha=0.5)
    plt.axvline(x=10.0, color='r', linestyle='--', label='Threshold (10°)')
    plt.xlabel('Rotation Error (degrees)')
    plt.ylabel('Stereo Inliers')
    plt.title('Relationship Between Rotation Error and Stereo Inliers')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('rotation_vs_stereo_inliers.jpg')


def output_transformation_stats(data):
    try:
        stats_df = pd.DataFrame({
            'Metric': ['Translation Error (m)', 'Rotation Error (deg)'],
            'Min': [data['translation_error'].min(), data['rotation_error_deg'].min()],
            'Max': [data['translation_error'].max(), data['rotation_error_deg'].max()],
            'Mean': [data['translation_error'].mean(), data['rotation_error_deg'].mean()],
            'Median': [data['translation_error'].median(), data['rotation_error_deg'].median()],
            'Good Ratio (%)': [
                (data['translation_error'] <= 1.0).mean() * 100,
                (data['rotation_error_deg'] <= 10.0).mean() * 100
            ]
        })

        stats_df.to_csv('transformation_error_stats.csv', index=False)

        # Create a more detailed text report
        with open('transformation_error_report.txt', 'w') as f:
            f.write("Transformation Error Analysis\n")
            f.write("===========================\n\n")
            f.write(f"Total data points: {len(data)}\n")

            # Safely access good_transformation column
            if 'good_transformation' in data.columns:
                good_count = len(data[data['good_transformation']])
                good_ratio = data['good_transformation'].mean() * 100
            else:
                good_count = 0
                good_ratio = 0.0

            f.write(
                f"Good transformations (trans ≤ 1.0m, rot ≤ 10°): {good_count}\n")
            f.write(f"Good transformation ratio: {good_ratio:.2f}%\n\n")

            f.write("Translation Error (meters)\n")
            f.write("------------------------\n")
            f.write(f"Min: {data['translation_error'].min():.4f}\n")
            f.write(f"Max: {data['translation_error'].max():.4f}\n")
            f.write(f"Mean: {data['translation_error'].mean():.4f}\n")
            f.write(f"Median: {data['translation_error'].median():.4f}\n")
            f.write(f"Within threshold (≤ 1.0m): {(data['translation_error'] <= 1.0).sum()} "
                    f"({(data['translation_error'] <= 1.0).mean() * 100:.2f}%)\n\n")

            f.write("Rotation Error (degrees)\n")
            f.write("----------------------\n")
            f.write(f"Min: {data['rotation_error_deg'].min():.4f}\n")
            f.write(f"Max: {data['rotation_error_deg'].max():.4f}\n")
            f.write(f"Mean: {data['rotation_error_deg'].mean():.4f}\n")
            f.write(f"Median: {data['rotation_error_deg'].median():.4f}\n")
            f.write(f"Within threshold (≤ 10°): {(data['rotation_error_deg'] <= 10.0).sum()} "
                    f"({(data['rotation_error_deg'] <= 10.0).mean() * 100:.2f}%)\n")
    except Exception as e:
        print(f"Error in output_transformation_stats: {str(e)}")

# Create a new function to rearrange columns in CSV loop closure results


def rearrange_loop_closure_csv(input_path, output_path=None):
    """
    Rearranges the columns in the loop closure CSV file to match their headers.

    Args:
        input_path: Path to the input CSV file
        output_path: Path to save the rearranged CSV file. If None, overwrites the input file.
    """
    try:
        if output_path is None:
            output_path = input_path

        # Read the CSV file
        df = pd.read_csv(input_path)

        # Define the expected column order based on file header
        expected_columns = [
            'Loop Closure Number', 'Robot 1', 'Relative Time 1', 'Robot 2', 'Relative Time 2',
            'Distance', 'Rotation Angle (Radian)', 'norm_bow_score', 'mono_inliers', 'stereo_inliers',
            'Estimated Distance', 'Estimated Angle(Radian)', 'Timestamp 1', 'Timestamp 2',
            'GT_Pose1_X', 'GT_Pose1_Y', 'GT_Pose1_Z', 'GT_Pose2_X', 'GT_Pose2_Y', 'GT_Pose2_Z',
            'Est_Pose1_X', 'Est_Pose1_Y', 'Est_Pose1_Z', 'Est_Pose2_X', 'Est_Pose2_Y', 'Est_Pose2_Z',
            'Relative Rotation Quaternion', 'Relative Translation Vector',
            'Estimated Relative Rotation', 'Estimated Relative Translation'
        ]

        # Select and reorder columns (only include columns that exist in dataframe)
        valid_columns = [col for col in expected_columns if col in df.columns]
        df_reordered = df[valid_columns]

        # Save the rearranged CSV
        df_reordered.to_csv(output_path, index=False)
        print(f"Rearranged CSV saved to {output_path}")
        return df_reordered

    except Exception as e:
        print(f"Error in rearrange_loop_closure_csv: {str(e)}")
        return None


def output_distance_ratios(data):
    """Calculate and output the ratio of distance categories to a file"""
    try:
        # Get total count
        total_count = len(data)

        # Count by distance category
        if 'distance_category' in data.columns:
            dist_counts = data['distance_category'].value_counts().sort_index()

            # Calculate ratios
            dist_ratios = dist_counts / total_count * 100

            # Create a DataFrame for the results
            result_df = pd.DataFrame({
                'Category': dist_counts.index,
                'Count': dist_counts.values,
                'Ratio (%)': dist_ratios.values.round(2)
            })

            # Save to file
            result_df.to_csv('distance_category_ratios.csv', index=False)

            # Also create a text file with more readable format
            with open('distance_category_ratios.txt', 'w') as f:
                f.write("Distance Category Distribution\n")
                f.write("-----------------------------\n")
                f.write(f"Total frames analyzed: {total_count}\n\n")

                for idx, row in result_df.iterrows():
                    f.write(
                        f"{row['Category']}: {int(row['Count'])} frames ({row['Ratio (%)']:.2f}%)\n")
        else:
            print("Warning: No distance_category column available for ratio analysis")
            with open('distance_category_ratios.txt', 'w') as f:
                f.write("Distance Category Distribution\n")
                f.write("-----------------------------\n")
                f.write("No distance category data available\n")
    except Exception as e:
        print(f"Error in output_distance_ratios: {str(e)}")
        with open('distance_category_ratios.txt', 'w') as f:
            f.write("Error generating distance category statistics\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Analyze inliers from CSV files in a directory')
    parser.add_argument('--directory', type=str, default='1207',
                        help='Directory containing CSV files')
    args = parser.parse_args()
    analyze_inliers(args.directory)
