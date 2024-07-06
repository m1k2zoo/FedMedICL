import torch
import numpy as np
from copy import deepcopy
import pandas as pd
import itertools
from dataset.util.AttributeGroup import AttributeGroup
import torchvision.transforms as transforms


class BaseDataset(torch.utils.data.Dataset):
    """
    Base class for custom datasets.

    This class abstracts the common functionalities needed across different datasets,
    including data loading, preprocessing, and handling of sensitive attributes.

    Attributes:
        dataframe (pandas.DataFrame): The original dataframe containing the dataset.
        path_to_dataset (str): The path to the dataset.
        sensitive_names (list): A list of sensitive attribute names.
        tasks_sensitive_name (str): A sensitive attribute name used for creating tasks.
        transform (torchvision.transforms): The data transformation to apply.
        targets (numpy.ndarray): The target labels of the dataset.
        task_id (int): The ID of the current task.
        client_id (int): The unique identifier for the client.
        all_attribute_groups (list): A list of all attribute groups.
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
        Initialize a new instance of the BaseDataset class.

        Args:
            dataframe (pandas.DataFrame): The original dataframe containing the dataset.
            path_to_dataset (str): The path to the dataset.
            sensitive_names (list): A list of sensitive attribute names.
            tasks_sensitive_name (str): A sensitive attribute name used for creating tasks.
            transform (torchvision.transforms): The data transformation to apply.
        """
        super(BaseDataset, self).__init__()

        self.dataframe = dataframe
        self.path_to_dataset = path_to_dataset
        self.sensitive_names = sensitive_names
        self.tasks_sensitive_name = tasks_sensitive_name
        self.transform = transform
        self.task_id = 0
        self.client_id = -1
        self.all_attribute_groups = []
        self.targets_set = []

        # TODO: Remove this and ensure that transform is always defined
        if self.transform == None:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            normalize = transforms.Normalize(mean=mean, std=std)

            transform_train = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

            self.transform = transform_train

    def shuffle_dataframe(self, seed):
        """
        Shuffle the DataFrame rows in-place, ensuring reproducibility using a specified seed.

        Args:
            seed (int): Seed value for random shuffling to ensure reproducibility.

        Returns:
            None: Modifies the DataFrame in-place and does not return any value.
        """
        self.dataframe = self.dataframe.sample(frac=1, random_state=seed).reset_index(drop=True)

    def assign_all_attribute_groups(self, all_attribute_groups):
        """
        Assign the list of all attribute groups.

        Args:
            all_attribute_groups (list): A list of all attribute groups.
        """
        self.all_attribute_groups = all_attribute_groups

    def assign_targets_set(self, targets_set):
        """
        Assign a set of unique targets in the full dataset.

        Args:
            targets_set (list): A list of all targets.
        """
        self.targets_set = targets_set

    def new_instance(self, dataframe):
        """
        Create and return an instance of the BaseDataset with a new dataframe.

        Args:
        dataframe (pandas.DataFrame): The new dataframe.

        Returns:
            tuple: A tuple containing the an updated instance of the BaseDataset.
        """
        instance = type(self)(
            dataframe=dataframe,
            path_to_dataset=self.path_to_dataset,
            sensitive_names=self.sensitive_names,
            tasks_sensitive_name=self.tasks_sensitive_name,
            transform=self.transform,
        )
        instance.assign_all_attribute_groups(self.all_attribute_groups)
        instance.assign_targets_set(self.targets_set)
        return instance

    def get_groups(self):
        """
        Get the list of all attribute groups.

        Returns:
            list: A list of all attribute groups.
        """
        return self.all_attribute_groups

    def get_item_group(self, item):
        """
        Get the group index for a specific item.

        Args:
            item: The item to get the group index for.

        Returns:
            int: The group index of the item.
        """
        group = [item[name] for name in self.sensitive_names]
        group = tuple(group)
        group_index = self.get_groups().index(group)
        return group_index

    def get_attribute_groups(self, attribute_type="group", is_sorted=True):
        """
        Get a list of AttributeGroup objects for each possible combination of attributes in the active dataframe.

        Args:
            attribute_type: The attribute to use for . Can be 'group', 'fine_grained_group', or 'target'

        Returns:
            list: A list of AttributeGroup objects.
        """
        attributeGroups = []
        if attribute_type not in ["group", "fine_grained_group", "target"]:
            raise ValueError(
                "attribute_type should be either 'group', 'fine_grained_group', or 'target'."
            )

        if attribute_type == "group":
            attribute_names = self.sensitive_names
        elif attribute_type == "fine_grained_group":
            attribute_names = [self.tasks_sensitive_name]
        elif attribute_type == "target":
            attribute_names = ["Target"]

        # Create a list of all possible combinations of attributes
        attributes = [self.dataframe[name].unique() for name in attribute_names]
        combinations = list(itertools.product(*attributes))

        # Define an `AttributeGroup` instance for each combination
        for combination in combinations:
            attributeGroups.append(AttributeGroup(attribute_names, combination, self.dataframe))

        # Sort the `attributeGroups` in descending order by size
        attributeGroups.sort(reverse=True)
        if is_sorted and self.all_attribute_groups != [] and attribute_type != "fine_grained_group":
            # Sort the `attributeGroups` to match the order of attributeGroups
            attributeGroups.sort(key=lambda x: self.all_attribute_groups.index(x.attribute_group))

        return attributeGroups

    def load_task(self, task_id, num_tasks, is_train, is_print):
        """
        Load a specific task for the BaseDataset.

        Args:
            task_id (int): The ID of the task to be loaded.
            num_tasks (int): Total number of tasks.
            is_train (bool): Whether the stream is for training or evaluation.
            is_print (bool): Whether to print information about the loaded task.

        Returns:
            None. Updates the dataframe, and task_id attributes by loading the specified task.

        Raises:
            AssertionError: If the given task_id is greater than or equal to the total number of tasks.

        """
        assert task_id < num_tasks, "Task ID must be smaller than the total number of tasks"

        self.task_id = task_id
        if num_tasks > 1:
            self.dataframe = self.tasks_dataframe[self.task_id]
        # TODO: Remove this if not needed
        # if is_train: # Train on the current task data
        #     self.dataframe = self.tasks_dataframe[self.task_id]
        # else: # Evaluate on all tasks up to and including the current one
        #     if num_tasks > 1:
        #         self.dataframe = pd.concat(self.tasks_dataframe[0:(self.task_id+1)])
        #     else:
        #         self.dataframe = self.tasks_dataframe[self.task_id]
        if is_print:
            print(
                f"Loading task: {task_id}, for client: {self.client_id}, split size: {self.dataframe.shape[0]}"
            )

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The size of the dataset.

        """
        return self.dataframe.shape[0]
