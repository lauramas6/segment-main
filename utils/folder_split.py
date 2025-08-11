import os
import shutil
import random
import argparse

def split_dataset(input_folder):
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folder {input_folder} not found.")

    parent_dir = os.path.dirname(input_folder)           # e.g., "pistachio"
    folder_name = os.path.basename(input_folder)         # e.g., "visible"

    # Create output dirs inside the parent folder
    train_dir = os.path.join(parent_dir, f"train_{folder_name}")
    val_dir = os.path.join(parent_dir, f"val_{folder_name}")
    test_dir = os.path.join(parent_dir, f"test_{folder_name}")
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Get all files in the folder (ignore directories)
    all_files = [f for f in os.listdir(input_folder) 
                 if os.path.isfile(os.path.join(input_folder, f))]

    random.shuffle(all_files)

    # Calculate split sizes
    total = len(all_files)
    train_count = int(total * 0.7)
    val_count = int(total * 0.2)

    train_files = all_files[:train_count]
    val_files = all_files[train_count:train_count + val_count]
    test_files = all_files[train_count + val_count:]

    # Move files into subfolders
    for f in train_files:
        shutil.move(os.path.join(input_folder, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.move(os.path.join(input_folder, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.move(os.path.join(input_folder, f), os.path.join(test_dir, f))

    print(f"Split complete for '{folder_name}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split folder into train/val/test sets with folder name suffix.")
    parser.add_argument("folder", help="Path to the folder containing files.")
    args = parser.parse_args()
    split_dataset(args.folder)
