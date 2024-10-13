import os
import pandas as pd
import shutil
from tqdm import tqdm

def copy_png_files(src_folder1, src_folder2, dest_folder):
    """
    Copies all .png files from two source folders to a destination folder.

    Parameters:
    src_folder1 (str): Path to the first source folder.
    src_folder2 (str): Path to the second source folder.
    dest_folder (str): Path to the destination folder.

    Returns:
    None
    """
    # Create destination folder if it doesn't exist
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Copy .png files from source folders to destination folder
    for src_folder in [src_folder1, src_folder2]:
        for file_name in tqdm(os.listdir(src_folder)):
            if file_name.endswith('.png'):
                shutil.copy(os.path.join(src_folder, file_name), dest_folder)

def combine_csv_files(src_folder1, src_folder2, dest_folder, dest_file_name):
    """
    Combines all .csv files from two source folders into a single .csv file in the destination folder.

    Parameters:
    src_folder1 (str): Path to the first source folder.
    src_folder2 (str): Path to the second source folder.
    dest_folder (str): Path to the destination folder.
    dest_file_name (str): Name of the destination .csv file.

    Returns:
    None
    """
    # Find .csv files in source folders
    csv_files = []
    for src_folder in [src_folder1, src_folder2]:
        for file_name in os.listdir(src_folder):
            if file_name.endswith('.csv'):
                csv_files.append(os.path.join(src_folder, file_name))

    # Read each .csv file into a DataFrame and concatenate them
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]
    combined_df = pd.concat(dfs, ignore_index=True)

    # Save the combined DataFrame to a .csv file in the destination folder
    combined_df.to_csv(os.path.join(dest_folder, dest_file_name), index=False)

# Main processing loop
for i in range(1, 6):
    for j in range(1, 6):
        if i <= j:
            continue
        for data_type in ["train", "test"]:
            """
            Processes datasets by copying .png files and combining .csv files from two source folders 
            into a destination folder.
            """
            # Define source and destination folders
            src_folder1 = f'{data_type}/dataset0{i}'
            src_folder2 = f'{data_type}/dataset0{j}'
            dest_folder = f'{data_type}/dataset0{i}x0{j}'
            
            if not os.path.exists(dest_folder):
                # Copy .png files
                copy_png_files(src_folder1, src_folder2, dest_folder)
            
            # Combine .csv files
            dest_file_name = 'label.csv'
            if not os.path.exists(os.path.join(dest_folder, dest_file_name)):
                combine_csv_files(src_folder1, src_folder2, dest_folder, dest_file_name)