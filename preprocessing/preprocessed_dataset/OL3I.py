import sys

[sys.path.append(i) for i in [".", ".."]]

from preprocess_helper import preprocess_metadata, process_age, process_sex, save_split
from PIL import Image
from tqdm import tqdm
import h5py
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
from os.path import dirname as up

pd.options.mode.chained_assignment = None  # default='warn'


def preprocess_data(data):
    """
    Preprocesses the input data (image) according to fit the right shape and pixel range.

    Args:
        data (numpy.ndarray): The input data array with shape (512, 512).

    Returns:
        numpy.ndarray: The preprocessed data array with shape (3, 512, 512).
    """
    # Clip the data to the range -1000 to 1000
    clipped_data = np.clip(data, -1000, 1000)

    # Reshape the data to a 3D array Shape: (3, 512, 512)
    reshaped_image = clipped_data.reshape(1, 512, 512)
    reshaped_data = np.repeat(reshaped_image, 3, axis=0)

    # Scale the pixel values to the range (0, 1)
    normalized_data = (reshaped_data - (-1000)) / (1000 - (-1000))

    # Scale the normalized values to the range (0, 255)
    scaled_data = normalized_data * 255

    # Round the values to integers
    rounded_data = np.round(scaled_data).astype(np.uint8)

    return rounded_data


def save_as_jpg(data, filename):
    """
    Saves the input data array as a JPEG image.

    Args:
        data (numpy.ndarray): The input data array with shape (3, 512, 512).
        filename (str): The filename to save the image as.

    Returns:
        None
    """
    # Convert the data array to a PIL Image
    image = Image.fromarray(np.transpose(data, (1, 2, 0)))

    # Save the image as a JPEG
    image.save(filename, "JPEG")


def prerpocess_ol3i_images(metadata_path, hdf5_path, image_path, image_id_col):
    """
    Preprocess OL3I images based on metadata and save them as JPEG files.

    Args:
        metadata_path (str): Path to the metadata file.
        hdf5_path (str): Path to the HDF5 data file.
        image_path (str): The path to store the images.
        image_id_col (str): The name of the column containing the image IDs.

    Returns:
        None
    """
    # Create the image_path directory if it does not exist
    Path(image_path).mkdir(parents=True, exist_ok=True)

    # Preprocess OL3I metadata
    metadata = pd.read_csv(metadata_path)
    print("\nThe preprocessed metadata includes following columns:\n", metadata.columns.values)

    # Open the HDF5 file
    with h5py.File(hdf5_path, "r") as hdf_file:
        # Loop over metadata rows and process images
        for i in tqdm(range(len(metadata))):
            anon_id = metadata.iloc[i][image_id_col]
            data = hdf_file[anon_id][()]

            # Preprocess the data to obtain the image
            image = preprocess_data(data)

            # Save the image as a JPEG file
            save_as_jpg(image, f"{image_path}/{anon_id}.jpg")

        print(f"\nProcessed and stored {i} images.\n")


def preprocess_OL3I():
    root_dir = up(up(os.getcwd()))
    config_path = os.path.join(root_dir, "configs", "datasets.json")
    with open(config_path, "r") as f:
        dataset_config = json.load(f)

    dataset_path = dataset_config["OL3I"]["path_to_dataset"]
    metadata_path = os.path.join(dataset_path, "clinical_data.csv")
    hdf5_path = os.path.join(dataset_path, "l3_slices.h5")
    selected_columns = ["anon_id", "bmi", "smoker", "age", "sex", "label_1y", "set_1y", "Path"]
    image_id_col = "anon_id"
    image_folder = "OL3I_images"
    image_path = os.path.join(dataset_path, image_folder)
    target_column = "label_1y"
    destination_path = os.path.join(dataset_path, "split")

    if not os.path.isdir(image_path):
        prerpocess_ol3i_images(metadata_path, hdf5_path, image_path, image_id_col)

    metadata = preprocess_metadata(
        dataset_path, metadata_path, selected_columns, image_id_col, image_folder, target_column
    )

    metadata = process_sex(metadata, "sex")
    metadata = process_age(metadata, "age")

    print("Final Columns:", metadata.columns.values)
    train_split = metadata[metadata["set_1y"] == "train"]
    val_split = metadata[metadata["set_1y"] == "val"]
    test_split = metadata[metadata["set_1y"] == "test"]
    save_split(train_split, destination_path, "train.csv", config_path, "OL3I", "train_meta_path")
    save_split(val_split, destination_path, "val.csv", config_path, "OL3I", "val_meta_path")
    save_split(test_split, destination_path, "test.csv", config_path, "OL3I", "test_meta_path")


if __name__ == "__main__":
    preprocess_OL3I()
