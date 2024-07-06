import sys

[sys.path.append(i) for i in [".", ".."]]

from preprocess_helper import (
    preprocess_metadata,
    process_age,
    process_sex,
    split_train_test,
    save_split,
)
import pandas as pd
import json
import os
from os.path import dirname as up


def combine_metadatas(dataset_path, selected_columns, image_folder, save_path):
    """
    Combines the metadata from the OD and OS CSV files and saves the selected columns to a new CSV file.

    Args:
    dataset_path (str): The path to the dataset directory.
    selected_columns (list): The list of column names to be selected in the metadata.
    image_folder (str): The directory where the images are stored.
    save_path (str): The path to save the combined CSV file.

    """
    # Read in the OD and OS metadata CSV files
    od_meta = pd.read_csv(
        os.path.join(dataset_path, "ClinicalData/patient_data_od.csv")
    )  # OD for right
    os_meta = pd.read_csv(
        os.path.join(dataset_path, "ClinicalData/patient_data_os.csv")
    )  # OS for left

    # Add the image paths for the OD and OS images
    os_meta["Path"] = [image_folder + "/RET" + x[1:] + "OS.jpg" for x in os_meta["ID"].values]
    od_meta["Path"] = [image_folder + "/RET" + x[1:] + "OD.jpg" for x in od_meta["ID"].values]

    # Combine the OD and OS metadata into a single DataFrame
    meta_all = pd.concat([od_meta, os_meta])

    # Select the relevant columns and save the metadata to a new CSV file
    meta_all = meta_all[selected_columns]
    meta_all.to_csv(save_path)


def preprocess_PAPILA():
    root_dir = up(up(os.getcwd()))
    config_path = os.path.join(root_dir, "configs", "datasets.json")
    with open(config_path, "r") as f:
        dataset_config = json.load(f)

    dataset_path = dataset_config["PAPILA"]["path_to_dataset"]
    metadata_path = os.path.join(dataset_path, "ClinicalData/patient_meta_concat.csv")
    selected_columns = ["ID", "Age", "Gender", "Diagnosis", "Path"]
    id_col = "ID"
    image_id_col = ""
    image_folder = "FundusImages"
    target_column = "Diagnosis"
    destination_path = os.path.join(dataset_path, "split")

    combine_metadatas(dataset_path, selected_columns, image_folder, metadata_path)

    metadata = preprocess_metadata(
        dataset_path, metadata_path, selected_columns, image_id_col, image_folder, target_column
    )

    metadata = process_sex(metadata, "Gender")
    metadata = process_age(metadata, "Age")

    print("Final Columns:", metadata.columns.values)
    train_split, other_split = split_train_test(metadata, id_col, test_size=0.3)
    val_split, test_split = split_train_test(other_split, id_col, test_size=0.5)
    save_split(train_split, destination_path, "train.csv", config_path, "PAPILA", "train_meta_path")
    save_split(val_split, destination_path, "val.csv", config_path, "PAPILA", "val_meta_path")
    save_split(test_split, destination_path, "test.csv", config_path, "PAPILA", "test_meta_path")


if __name__ == "__main__":
    preprocess_PAPILA()
