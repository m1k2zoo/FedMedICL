import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import os
import json
import re

pd.options.mode.chained_assignment = None  # default='warn'


def add_image_paths(df, image_id_col, image_folder, add_extention=True):
    """
    Add image paths to the metadata DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the metadata.
        image_id_col (str): The name of the column containing the image IDs.
        image_folder (str): The directory where the images are stored.
        add_extention (bool, optional): Whether to manually add image extensions. Defaults to True.

    Returns:
        pd.DataFrame: The modified DataFrame with the added 'Path' column.
    """
    pathlist = df[image_id_col].values.tolist()
    if add_extention:
        paths = [f"{image_folder}/{i}.jpg" for i in pathlist]
    else:
        paths = [f"{image_folder}/{i}" for i in pathlist]
    df["Path"] = paths

    return df


def unify_sex_values(df, sex_column, new_column_name):
    """
    Unifies the values in the specified sex column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the sex column.
        sex_column (str): The name of the sex column.
        new_column_name (str): The name of the new column to be created.

    Returns:
        pd.DataFrame: The DataFrame with unified sex values.
    """
    df[new_column_name] = df[sex_column].replace({"male": "M", "female": "F"})
    df[new_column_name] = df[new_column_name].replace({"Male": "M", "Female": "F"})
    df[new_column_name] = df[new_column_name].replace({0: "M", 1: "F"})

    return df


def split_age_groups(df, age_column, new_column_name, age_bins, group_labels):
    """
    Splits the subjects into different age groups based on the age column.

    Args:
        df (pd.DataFrame): The DataFrame containing the age column.
        age_column (str): The name of the age column.
        new_column_name (str): The name of the new column to be created.
        age_bins (list): The bin boundaries for age grouping.
        group_labels (list): The labels for the age groups.

    Returns:
        pd.DataFrame: The DataFrame with the new age group column.
    """
    df[new_column_name] = pd.cut(df[age_column], bins=age_bins, labels=group_labels, right=False)
    return df


def process_age(df, age_column):
    """
    Process the age column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the age column.
        age_column (str): The name of the age column.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    # Check for non-numeric values in the 'age' column
    df[age_column] = pd.to_numeric(df[age_column], errors="coerce")
    df = df.dropna(subset=[age_column])

    # Create the 'Age_multi' column
    df = split_age_groups(
        df,
        age_column,
        "Age_multi",
        age_bins=[-1, 19, 39, 59, 79, float("inf")],
        group_labels=[0, 1, 2, 3, 4],
    )

    # Create the 'Age_binary' column
    df = split_age_groups(
        df, age_column, "Age_binary", age_bins=[-1, 59, float("inf")], group_labels=[0, 1]
    )

    # Drop the original age column
    df.drop([age_column], axis=1, inplace=True)
    return df


def process_sex(df, sex_column):
    """
    Process the sex column of the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the sex column.
        sex_column (str): The name of the sex column.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """

    # Drop rows with invalid sex values
    df = df[~(df[sex_column] == "unknown")]
    df = df[~(df[sex_column] == "Unknown")]
    df = df.dropna(subset=[sex_column])

    # Unify sex values
    df = unify_sex_values(df, sex_column, "Sex")

    # Drop the original sex column
    if sex_column != "Sex":
        df.drop([sex_column], axis=1, inplace=True)
    return df


def preprocess_metadata(
    dataset_path,
    metadata_path,
    selected_columns,
    image_id_col,
    image_folder,
    target_column,
    remove_missing_images=True,
    target_to_categorical=True,
    add_extention=True,
):
    """
    Preprocesses the dataset metadata.

    Args:
        dataset_path (str): The path to the dataset directory.
        metadata_path (str): The path to the metadata file.
        selected_columns (list): The list of column names to be selected in the metadata.
        image_id_col (str): The name of the column containing the image IDs.
        image_folder (str): The directory where the images are stored.
        target_column (str): The name of the column containing the target variable.
        remove_missing_images (bool, optional): Whether to remove rows with missing images. Defaults to True.
        target_to_categorical (bool, optional): Whether to convert the target variable to a categorical variable. Defaults to True.
        add_extention (bool, optional): Whether to manually add image extensions. Defaults to True.
    Returns:
        pd.DataFrame: The preprocessed metadata DataFrame.
    """
    # Step 1: Read the metadata into a DataFrame
    metadata = pd.read_csv(metadata_path)

    # Step 2: Add image path to the metadata
    if "Path" not in metadata:
        metadata = add_image_paths(metadata, image_id_col, image_folder, add_extention)
    print("Initial columns (first 10):", metadata.columns.values[:10])

    # Step 3: Remove rows with missing image paths
    if remove_missing_images:
        invalid_paths = []
        for i, row in metadata.iterrows():
            if not os.path.exists(os.path.join(dataset_path, row["Path"])):
                invalid_paths.append(i)
        metadata.drop(invalid_paths, inplace=True)
        print(f"Removed {len(invalid_paths)} rows with missing images.")

    # Step 4: Select the relevant columns and remove the rest
    metadata = metadata[selected_columns]

    # Step 5: Copy target column to 'target'
    metadata["Target"] = metadata[target_column]
    if target_to_categorical and not is_numeric_dtype(metadata["Target"]):
        metadata["Target"] = pd.Categorical(metadata["Target"]).codes

    metadata.drop([target_column], axis=1, inplace=True)

    return metadata


def split_train_test(df, id_col, test_size=0.1):
    """
    Splits the DataFrame into train and test sets.

    Args:
        df (pd.DataFrame): The DataFrame containing the metadata.
        id_col (str): The name of the column containing the IDs.
        test_size (float): Ratio of the test split size.

    Returns:
        pd.DataFrame: The training and testing metadata DataFrames.
    """
    # Get IDs
    ids = np.unique(df[id_col])

    # Split the unique image IDs into train and test sets
    # Using a SEED of 0 for reproducibility
    sub_train, sub_test = train_test_split(ids, test_size=test_size, random_state=0)

    # Get the metadata for the train and test sets
    train_meta = df[df[id_col].isin(sub_train)]
    test_meta = df[df[id_col].isin(sub_test)]

    return train_meta, test_meta


def save_split(df, destination_path, filename, config_path, dataset_key, split_key):
    """
    Saves a DataFrame as CSV files.

    Args:
        df (pd.DataFrame): The metadata DataFrame.
        destination_path (str): The directory where the CSV file will be saved.
        filename (str): The name of the CSV file.
        config_path (str): Path to the configuration JSON file.
        dataset_key (str): Key for the dataset in the configuration JSON.
        split_key (str): Key for the split type (train, val, test) in the configuration JSON.
    """
    # Create the destination_path directory if it does not exist
    Path(destination_path).mkdir(parents=True, exist_ok=True)

    # Save the df as CSV files
    full_path = f"{destination_path}/{filename}"
    df.to_csv(full_path, index=False)
    print(f"Saving split at: {full_path}")

    # Update the configuration JSON file
    with open(config_path, "r+") as file:
        config = json.load(file)
        config[dataset_key][split_key] = full_path
        file.seek(0)
        json.dump(config, file, indent=4)
        file.truncate()

    # Post-processing the JSON to modify specific list formatting in JSON file
    with open(config_path, "r+") as file:
        content = file.read()
        # Adjust the regex to correctly format multiline arrays into a single line
        content = re.sub(
            r"\[\s*((?:\[\s*[\d.]+,\s*[\d.]+\s*\]\s*,?\s*)+)\]",
            lambda m: "[" + m.group(1).replace("\n", "").replace(" ", "").replace(",", ", ") + "]",
            content,
        )
        file.seek(0)
        file.write(content)
        file.truncate()

    config_file_name = os.path.basename(config_path)
    path_to_csv = os.path.basename(full_path)
    print(f"Updated the key '{split_key}' in '{config_file_name}' to the path of {path_to_csv}\n")
