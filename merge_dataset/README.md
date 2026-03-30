# HM3D Dataset Merge Script

This document describes the `merge_dataset.py` utility, which is used to prepare the Habitat-Matterport 3D (HM3D) dataset.

## Overview

The `merge_dataset.py` script is a utility designed to combine different components of the HM3D dataset into a single, unified directory. It assumes you have downloaded separate parts of the dataset (semantic annotations, 3D geometry models, and configuration files) and organizes them into one cohesive folder structure.

The script performs the following actions:
1.  Identifies a list of all "house" scenes from the semantic annotations folder.
2.  For each house, it creates a corresponding folder in a new `hm3d-train-complete` directory.
3.  It copies both the semantic annotation files and the 3D geometry files (`.glb`) for each house into its respective folder within `hm3d-train-complete`.
4.  Finally, it copies the master scene dataset configuration file into the root of the new `hm3d-train-complete` directory.

## Prerequisite Folder Structure

Before running the script, you must ensure your dataset files are organized in a specific way. The script requires the following structure within the main `nav_gen` project:

```
nav_gen/
├── data/
│   └── hm3d/
│       └── train/
│           ├── hm3d-train-semantic-annots-v0.2/  <-- SOURCE 1
│           │   ├── 00006-HkseAnWCgqk/
│           │   └── ...
│           │
│           ├── hm3d-train-habitat-v0.2/        <-- SOURCE 2
│           │   ├── 00006-HkseAnWCgqk/
│           │   └── ...
│           │
│           └── hm3d-train-semantic-configs-v0.2/ <-- SOURCE 3
│               └── hm3d_annotated_train_basis.scene_dataset_config.json
│
└── utils/
    └── merge_dataset/
        └── merge_dataset.py  <-- THIS SCRIPT
```

## Usage

To run this script, first navigate to its directory and then execute it with Python.

```bash
# Navigate to the script's directory
cd utils/merge_dataset

# Run the script
python merge_dataset.py
```
**Note:** This usage instruction assumes the `base_data_path` inside the script is correctly set to `../../data/hm3d/train` to navigate from the script's location back to the project root and into the `data` folder.

## Output Structure

After the script successfully completes, you will have a new directory named `hm3d-train-complete` inside `nav_gen/data/hm3d/train/`. This folder will contain the merged dataset, ready for use by other scripts in the project.

The final structure will look like this:

```
nav_gen/
└── data/
    └── hm3d/
        └── train/
            └── hm3d-train-complete/  <-- DESTINATION
                ├── 00006-HkseAnWCgqk/
                │   ├── ... (semantic files)
                │   └── ... (geometry .glb file)
                ├── ... (other merged houses)
                └── hm3d_annotated_train_basis.scene_dataset_config.json
```
