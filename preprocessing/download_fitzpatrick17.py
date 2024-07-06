import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

import pandas as pd
import json
from os.path import dirname as up


def download_image(destination_path, image_url, md5_hash, error_list):
    """
    Download an image from the specified URL and save it to the given path.

    Args:
        destination_path (str): The destination directory where the downloaded images will be saved.
        image_url (str): The URL of the image to download.
        md5_hash (str): The MD5 hash of the image used for constructing the image path.
        error_list (list): A list to store error information for images that fail to download.

    Returns:
        str: The path of the downloaded image if successful, or an error message if an exception occurs.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        image_path = f"{destination_path}/{md5_hash}.jpg"
        with open(image_path, "wb") as file:
            file.write(response.content)
        return image_path
    except requests.exceptions.RequestException as e:
        error_list.append((md5_hash, image_url))
        return f"Error downloading image: {md5_hash}\nError message: {e}"


def download_images_parallel(annot_data, destination_path, num_workers=5):
    """
    Download images in parallel from the URLs specified in the annotation data.

    Args:
        annot_data (pd.DataFrame): The DataFrame containing annotation data with 'url' and 'md5hash' columns.
        destination_path (str): The destination directory where the downloaded images will be saved.
        num_workers (int): The number of parallel workers to use for downloading. Default is 5.

    Returns:
        None
    """
    # Create the destination_path directory if it doesn't exist
    os.makedirs(destination_path, exist_ok=True)

    error_list = []

    # Create a ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []

        for index, row in annot_data.iterrows():
            image_url = row["url"]
            md5_hash = row["md5hash"]

            # Submit the download task to the executor
            future = executor.submit(
                download_image, destination_path, image_url, md5_hash, error_list
            )
            futures.append(future)

        # Wait for all tasks to complete
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, str):
                print(result)
            else:
                print(f"Image downloaded: {result}")

    # Save error_list to a CSV file
    with open("error_list.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["md5_hash", "image_url"])
        writer.writerows(error_list)


# read metadata
root_dir = up(os.getcwd())
with open(root_dir + "/configs/datasets.json", "r") as f:
    dataset_config = json.load(f)

path = dataset_config["fitzpatrick17k"]["path_to_dataset"]
try:
    csv_path = os.path.join(path, "fitzpatrick17k.csv")
    annot_data = pd.read_csv(csv_path)
except FileNotFoundError:
    raise FileNotFoundError(
        "The file 'fitzpatrick17k.csv' does not exist in the specified directory: {}. "
        "Please download it from https://github.com/mattgroh/fitzpatrick17k and ensure it is in the correct path.".format(
            path
        )
    )


destination_path = os.path.join(path, "images")

# Call the function to download images in parallel
download_images_parallel(annot_data, destination_path, num_workers=6)
