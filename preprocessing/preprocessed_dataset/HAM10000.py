import sys

[sys.path.append(i) for i in [".", ".."]]

from preprocess_helper import (
    preprocess_metadata,
    process_age,
    process_sex,
    split_train_test,
    save_split,
)
import json
import os
from os.path import dirname as up


def preprocess_HAM10000():
    root_dir = up(up(os.getcwd()))
    config_path = os.path.join(root_dir, "configs", "datasets.json")
    with open(config_path, "r") as f:
        dataset_config = json.load(f)

    dataset_path = dataset_config["HAM10000"]["path_to_dataset"]
    metadata_path = os.path.join(dataset_path, "HAM10000_metadata.csv")
    selected_columns = ["lesion_id", "image_id", "dx", "age", "sex", "Path"]
    id_col = "lesion_id"
    image_id_col = "image_id"
    image_folder = "HAM10000_images"
    target_column = "dx"
    destination_path = os.path.join(dataset_path, "split")

    metadata = preprocess_metadata(
        dataset_path, metadata_path, selected_columns, image_id_col, image_folder, target_column
    )

    metadata = process_sex(metadata, "sex")
    metadata = process_age(metadata, "age")

    print("Final Columns:", metadata.columns.values)
    train_split, other_split = split_train_test(metadata, id_col, test_size=0.2)
    val_split, test_split = split_train_test(other_split, id_col, test_size=0.5)
    save_split(
        train_split, destination_path, "train.csv", config_path, "HAM10000", "train_meta_path"
    )
    save_split(val_split, destination_path, "val.csv", config_path, "HAM10000", "val_meta_path")
    save_split(test_split, destination_path, "test.csv", config_path, "HAM10000", "test_meta_path")


if __name__ == "__main__":
    preprocess_HAM10000()
