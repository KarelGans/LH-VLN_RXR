import os
import shutil
import sys

def merge_datasets():
    # --- Configuration ---
    # We are running from: ~/Karel_Programs/nav_gen_train_val_dataset/nav_gen
    base_data_path = "data/hm3d/val_temp"
    
    # Source Folders
    src_semantic = os.path.join(base_data_path, "hm3d-val-semantic-annots-v0.2")
    src_geometry = os.path.join(base_data_path, "hm3d-val-habitat-v0.2")
    src_config   = os.path.join(base_data_path, "hm3d-val-semantic-configs-v0.2")
    
    # Destination Folder
    dst_complete = os.path.join(base_data_path, "hm3d-val-complete-2")

    # --- Validation ---
    if not os.path.exists(src_semantic):
        print(f"Error: Source folder not found: {src_semantic}")
        sys.exit(1)
    if not os.path.exists(src_geometry):
        print(f"Error: Source folder not found: {src_geometry}")
        sys.exit(1)

    # Create Destination
    if not os.path.exists(dst_complete):
        print(f"Creating new directory: {dst_complete}")
        os.makedirs(dst_complete)
    else:
        print(f"Warning: Destination directory already exists: {dst_complete}")
        print("New files will be merged into it.")

    # --- Step 1: Get list of houses ---
    # We use the semantic folder as the 'master list'
    house_folders = [f for f in os.listdir(src_semantic) if os.path.isdir(os.path.join(src_semantic, f))]
    total = len(house_folders)
    
    print(f"Found {total} houses to process...")

    # --- Step 2: Loop and Merge ---
    for i, house_id in enumerate(house_folders):
        print(f"[{i+1}/{total}] Merging {house_id}...", end="\r")

        # Define paths for this specific house
        path_sem = os.path.join(src_semantic, house_id)
        path_geo = os.path.join(src_geometry, house_id)
        path_dst = os.path.join(dst_complete, house_id)

        # A. Copy Semantic Data (Creates the folder if needed)
        # dirs_exist_ok=True allows us to merge into existing folders
        shutil.copytree(path_sem, path_dst, dirs_exist_ok=True)

        # B. Copy Geometry Data (Merges into the same folder)
        if os.path.exists(path_geo):
            shutil.copytree(path_geo, path_dst, dirs_exist_ok=True)
        else:
            print(f"\nWarning: Geometry missing for {house_id}")

    print(f"\n\nHouse data merge complete.")

    # --- Step 3: Copy the Master Config JSON ---
    # We look for the .scene_dataset_config.json file in the config folder
    print("Copying Master Scene Dataset Config...")
    config_file_name = "hm3d_annotated_train_basis.scene_dataset_config.json"
    src_config_file = os.path.join(src_config, config_file_name)
    dst_config_file = os.path.join(dst_complete, config_file_name)

    if os.path.exists(src_config_file):
        shutil.copy2(src_config_file, dst_config_file)
        print(f"Success! Config copied to: {dst_config_file}")
    else:
        print(f"Error: Could not find config file at {src_config_file}")
        print("You may need to manually copy the .json file to the root of hm3d-train-complete.")

    print("\n---------------------------------------------------")
    print("MERGE FINISHED.")
    print(f"New Dataset Path: {os.path.abspath(dst_complete)}")
    print(f"New Config Path:  {os.path.abspath(dst_config_file)}")
    print("---------------------------------------------------")

if __name__ == "__main__":
    merge_datasets()