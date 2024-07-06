import sys

[sys.path.append(i) for i in [".", ".."]]

from preprocess_helper import preprocess_metadata, split_train_test, save_split
import json
import os
from os.path import dirname as up


def preprocess_fitzpatrick17k():
    root_dir = up(up(os.getcwd()))
    config_path = os.path.join(root_dir, "configs", "datasets.json")
    with open(config_path, "r") as f:
        dataset_config = json.load(f)

    dataset_path = dataset_config["fitzpatrick17k"]["path_to_dataset"]
    metadata_path = os.path.join(dataset_path, "fitzpatrick17k.csv")
    selected_columns = ["md5hash", "fitzpatrick_scale", "three_partition_label", "Path"]
    image_id_col = "md5hash"
    image_folder = "images"
    target_column = "three_partition_label"
    destination_path = os.path.join(dataset_path, "split")

    metadata = preprocess_metadata(
        dataset_path, metadata_path, selected_columns, image_id_col, image_folder, target_column
    )

    # Remove skin type == null
    metadata = metadata[metadata["fitzpatrick_scale"] != -1]

    # Define skin_type
    metadata["skin_type"] = metadata["fitzpatrick_scale"] - 1

    # Convert skin_type to binary
    skin_lists = metadata["skin_type"].values.tolist()
    metadata["skin_binary"] = [0 if x <= 2 else 1 for x in skin_lists]
    metadata.drop(["fitzpatrick_scale"], axis=1, inplace=True)

    print("Final Columns:", metadata.columns.values)
    train_split, other_split = split_train_test(metadata, image_id_col, test_size=0.2)
    val_split, test_split = split_train_test(other_split, image_id_col, test_size=0.5)
    save_split(
        train_split, destination_path, "train.csv", config_path, "fitzpatrick17k", "train_meta_path"
    )
    save_split(
        val_split, destination_path, "val.csv", config_path, "fitzpatrick17k", "val_meta_path"
    )
    save_split(
        test_split, destination_path, "test.csv", config_path, "fitzpatrick17k", "test_meta_path"
    )


if __name__ == "__main__":
    preprocess_fitzpatrick17k()
