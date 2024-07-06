from PIL import Image
from dataset.BaseDataset import BaseDataset
import os


class Dataset2D(BaseDataset):
    """
    A custom dataset class for handling 2D image datasets with sensitive attributes.

    Attributes:
        dataframe (pd.DataFrame): DataFrame containing metadata for the dataset.
        path_to_dataset (str): Path to the directory containing the dataset.
        sensitive_names (list): List of sensitive attribute names.
        tasks_sensitive_name (str): A sensitive attribute name used for creating tasks.
        transform (callable): Transform to be applied to the images.

    """

    def __init__(
        self,
        dataframe,
        path_to_dataset,
        sensitive_names,
        tasks_sensitive_name,
        transform,
    ):
        """
        Initializes the Dataset2D with dataset-specific details and transformations applicable to image data.

        Args:
            dataframe (pd.DataFrame): DataFrame containing metadata for the dataset.
            path_to_dataset (str): Path to the directory containing the dataset.
            sensitive_names (list): List of sensitive attribute names.
            tasks_sensitive_name (str): A sensitive attribute name used for creating tasks.
            transform (callable): Transform to be applied to the images.
        """
        super(Dataset2D, self).__init__(
            dataframe, path_to_dataset, sensitive_names, tasks_sensitive_name, transform
        )

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding target label and sensitive attributes based on the provided index.

        Args:
            idx (int): Index of the data item.

        Returns:
            tuple: A tuple containing the transformed image, its corresponding target label, and the index of its sensitive group.
        """
        item = self.dataframe.iloc[idx]

        image_path = os.path.join(self.path_to_dataset, item["Path"])
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)

        target = item["Target"]
        sensitive = self.get_item_group(item)

        return img, target, sensitive

    def get_labels(self):
        """
        Retrieves all labels (targets) from the dataset.

        Returns:
            pd.Series: A Series containing all the target labels from the dataframe.
        """
        return self.dataframe["Target"]
