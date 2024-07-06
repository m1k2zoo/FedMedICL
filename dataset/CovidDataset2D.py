import os
from PIL import Image
from dataset.Dataset2D import Dataset2D


class CovidDataset2D(Dataset2D):
    """
    A specialized dataset class for handling 2D medical image datasets, particularly CheXpert and COVID.
    It extends Dataset2D to manage data from two distinct dataset sources with separate paths.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing the dataset metadata.
        paths_to_dataset (dict): A dictionary of paths to the datasets with keys 'COVID' and 'CheXpert'.
        sensitive_names (list): List of sensitive attribute names.
        tasks_sensitive_name (str): A sensitive attribute name used for creating tasks.
        transform (callable): Transform to be applied to the images.
    """

    def __init__(
        self,
        dataframe,
        paths_to_dataset,
        sensitive_names,
        tasks_sensitive_name,
        transform,
    ):
        """
        Initializes the CovidDataset2D with specific paths for CheXpert and COVID datasets.

        Args:
            dataframe (pd.DataFrame): DataFrame containing the dataset metadata.
            paths_to_dataset (dict): A dictionary of paths to the datasets with keys 'COVID' and 'CheXpert'.
            sensitive_names (list): List of sensitive attribute names.
            tasks_sensitive_name (str): A sensitive attribute name used for creating tasks.
            transform (callable): Trans form to be applied to the images.
        """

        # Initialize the parent class with one of the dataset paths as a placeholder
        super().__init__(
            dataframe,
            list(paths_to_dataset.values())[0],  # Placeholder
            sensitive_names,
            tasks_sensitive_name,
            transform,
        )
        self.paths_to_dataset = paths_to_dataset

    def __getitem__(self, idx):
        """
        Fetches the image and its labels at the index `idx` by determining the correct dataset path from the metadata.

        Args:
            idx (int): The index of the data item to fetch.

        Returns:
            tuple: A tuple containing the transformed image, target label, and sensitive group index.
        """
        item = self.dataframe.iloc[idx]

        # Determine the dataset path based on the "Path" field in the DataFrame
        if "CheXpert" in item["Path"].split("/")[0]:
            # CheXpert dataset
            dataset_path = self.paths_to_dataset["CheXpert"]
        else:
            # COVID dataset
            dataset_path = self.paths_to_dataset["COVID"]

        image_path = os.path.join(dataset_path, item["Path"])
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        target = item["Target"]
        sensitive = self.get_item_group(item)

        return img, target, sensitive

    # Overwrite BaseDataset implementation to support a dictionary of paths_to_dataset
    def new_instance(self, dataframe):
        """
        Create and return an instance of the BaseDataset with a new dataframe.

        Args:
            dataframe (pandas.DataFrame): The new dataframe.

        Returns:
            CovidDataset2D: An instance of CovidDataset2D initialized with the new DataFrame.
        """
        instance = type(self)(
            dataframe=dataframe,
            paths_to_dataset=self.paths_to_dataset,
            sensitive_names=self.sensitive_names,
            tasks_sensitive_name=self.tasks_sensitive_name,
            transform=self.transform,
        )
        instance.assign_all_attribute_groups(self.all_attribute_groups)
        instance.assign_targets_set(self.targets_set)
        return instance
