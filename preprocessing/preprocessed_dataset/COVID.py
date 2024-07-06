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


def preprocess_COVID():
    root_dir = up(up(os.getcwd()))
    config_path = os.path.join(root_dir, "configs", "datasets.json")
    with open(config_path, "r") as f:
        dataset_config = json.load(f)

    dataset_path = dataset_config["COVID"]["path_to_dataset"]
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    selected_columns = ["Path", "sex", "age", "view", "finding", "patientid"]

    id_col = "PATIENT"
    image_id_col = "filename"
    image_folder = "images"
    target_column = "finding"
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
        add_extention=False,
    )

    metadata = process_sex(metadata, "sex")
    metadata = process_age(metadata, "age")

    # Only select patients with COVID-19
    metadata = metadata[metadata["Target"] == "Pneumonia/Viral/COVID-19"]

    # Assign COVID as label 2, given that it will be merged with CheXpert which has labels 0 and 1
    metadata.Target = 2

    # Only keep samples with x-ray images of the views: ["PA", "AP", "AP Supine"]
    metadata = metadata[metadata["view"].isin(["PA", "AP", "AP Supine"])]

    # Use similar names to CheXpert columns
    metadata = metadata.rename(columns={"view": "AP/PA"})
    metadata = metadata.rename(columns={"patientid": "PATIENT"})
    metadata["PRIMARY_RACE"] = "Unknown"

    # Use CheXpert columns order
    CheXpert_columns_order = [
        "Path",
        "Sex",
        "AP/PA",
        "Target",
        "Age_multi",
        "Age_binary",
        "PATIENT",
        "PRIMARY_RACE",
    ]
    metadata = metadata[CheXpert_columns_order]

    print("Final Columns:", metadata.columns.values)
    train_split, other_split = split_train_test(metadata, id_col, test_size=0.2)
    val_split, test_split = split_train_test(other_split, id_col, test_size=0.5)
    save_split(train_split, destination_path, "train.csv", config_path, "COVID", "train_meta_path")
    save_split(val_split, destination_path, "val.csv", config_path, "COVID", "val_meta_path")
    save_split(test_split, destination_path, "test.csv", config_path, "COVID", "test_meta_path")


if __name__ == "__main__":
    preprocess_COVID()
