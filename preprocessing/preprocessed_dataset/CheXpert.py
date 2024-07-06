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
import re
import json
import os
from os.path import dirname as up


def extract_patient_id(path):
    """
    Extracts the patient ID from the given path string.

    Args:
        path (str): The path string containing the patient ID.

    Returns:
        str: The extracted patient ID.
    """
    # Regular expression pattern to match 'patient' followed by one or more digits
    pattern = r"patient(\d+)"
    match = re.search(pattern, path)
    if match:
        return match.group(0)  # Return the entire matched substring (e.g. "patient<ID>")
    else:
        return None  # Return None if no patient ID is found in the path


def convert_no_finding_label(label):
    """
    Converts the 'No Finding' label to numeric values.

    Args:
        label (object): The label value.

    Returns:
        int: The corresponding numeric value.
    """
    if label == 1.0:
        return 1
    else:
        return 0
    # elif label == 0.0:
    #     return 0
    # else:
    #     return -1  # Handle other cases if applicable


def preprocess_CheXpert():
    root_dir = up(up(os.getcwd()))
    config_path = os.path.join(root_dir, "configs", "datasets.json")
    with open(config_path, "r") as f:
        dataset_config = json.load(f)

    dataset_path = dataset_config["CheXpert"]["path_to_dataset"]

    metadata_path = os.path.join(dataset_path, "train_cheXbert.csv")
    metadata_race_path = os.path.join(dataset_path, "CHEXPERT DEMO.csv")
    selected_columns = ["Path", "Sex", "Age", "AP/PA", "No Finding"]

    id_col = "PATIENT"
    image_id_col = ""
    image_folder = ""
    target_column = "No Finding"
    destination_path = os.path.join(dataset_path, "split")

    metadata = preprocess_metadata(
        dataset_path,
        metadata_path,
        selected_columns,
        image_id_col,
        image_folder,
        target_column,
        remove_missing_images=False,
        target_to_categorical=False,
    )

    metadata = process_sex(metadata, "Sex")
    metadata = process_age(metadata, "Age")

    # Create a new column 'PATIENT' in the metadata DataFrame
    metadata["PATIENT"] = metadata["Path"].apply(extract_patient_id)

    # Read race metadata
    race_data = pd.read_csv(metadata_race_path)

    # Create a new column 'PRIMARY_RACE' by merging with race_data based on 'PATIENT' column
    all_metadata = pd.merge(
        metadata, race_data[["PATIENT", "PRIMARY_RACE"]], on="PATIENT", how="left"
    )

    # Convert 'Target' to numeric values
    all_metadata["Target"] = all_metadata["Target"].map(convert_no_finding_label)

    print("Final Columns:", metadata.columns.values)
    train_split, other_split = split_train_test(all_metadata, id_col, test_size=0.2)
    val_split, test_split = split_train_test(other_split, id_col, test_size=0.5)
    save_split(
        train_split, destination_path, "train.csv", config_path, "CheXpert", "train_meta_path"
    )
    save_split(val_split, destination_path, "val.csv", config_path, "CheXpert", "val_meta_path")
    save_split(test_split, destination_path, "test.csv", config_path, "CheXpert", "test_meta_path")


if __name__ == "__main__":
    preprocess_CheXpert()
