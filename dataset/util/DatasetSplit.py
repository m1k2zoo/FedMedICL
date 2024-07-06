import warnings
import pandas as pd
import numpy as np
from copy import deepcopy
import random


class DatasetSplit:
    """
    Manages dataset splits for training, validation, testing, and holdout sets in federated learning scenarios. This class
    is designed to handle the distribution of datasets among multiple clients, supporting complex scenarios like
    imbalanced data distribution and novel disease tasks.

    Each client is assigned a DatasetSplit in the initalizaiton phase.

    Attributes:
        name (str): Name of the dataset used for identification.
        train_set (BaseDataset): Dataset split used for training.
        val_set (BaseDataset): Dataset split used for validation.
        test_set (BaseDataset): Dataset split used for testing.
        holdout_set (BaseDataset): Copy of the test set used for holdout scenarios.
        num_classes (int): Total number of unique classes in the dataset.
        client_id (int): Identifier for the client owning this split.
        num_tasks (int): Number of tasks into which the dataset is divided for training.
        all_attribute_groups (list): List of all attribute groups to maintain consistency across splits.
        targets_set (list): List of all unique target labels across the dataset.
    """

    def __init__(self, name, train_set, val_set, test_set, num_classes, imbalance_type):
        """
        Initializes a new instance of the DatasetSplit class with specified datasets and configuration.

        Args:
            name (str): The name of the dataset.
            train_set (BaseDataset): The training dataset.
            val_set (BaseDataset): The validation dataset.
            test_set (BaseDataset): The testing dataset.
            num_classes (int): The number of unique classes in the dataset.
            imbalance_type (str): Type of imbalance to apply to the dataset splits, if any.
        """
        self.name = name
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.holdout_set = deepcopy(test_set)
        self.num_classes = num_classes
        self.num_tasks = 1
        self.client_id = -1

        # Define a common attribute groups for all splits
        # This is useful to have the same attribute groups across the clients splits
        self.all_attribute_groups = [
            group.attribute_group for group in train_set.get_attribute_groups(imbalance_type)
        ]
        self.train_set.assign_all_attribute_groups(self.all_attribute_groups)
        self.val_set.assign_all_attribute_groups(self.all_attribute_groups)
        self.test_set.assign_all_attribute_groups(self.all_attribute_groups)
        self.holdout_set.assign_all_attribute_groups(self.all_attribute_groups)

        self.num_groups = len(self.all_attribute_groups)

        # Define a common list of target set
        self.targets_set = (
            self.train_set.dataframe.Target.value_counts().sort_index().index.tolist()
        )
        self.train_set.assign_targets_set(self.targets_set)
        self.val_set.assign_targets_set(self.targets_set)
        self.test_set.assign_targets_set(self.targets_set)
        self.holdout_set.assign_targets_set(self.targets_set)

    def assign_train_set(self, train_dataframe):
        """
        Assigns a new train dataframe to the DatasetSplit instance and updates attribute groups.

        Args:
            new_train_set (Dataframe): The new train set to assign.

        Returns:
            None. Updates the train_set attribute and sets attribute groups of the new train set.

        """
        self.train_set = self.train_set.new_instance(train_dataframe)

    def create_tasks(self, num_tasks, seed, split_type, split_ratios=None):
        """
        Create tasks for continual learning by splitting the dataset according to the specified strategy.

        Args:
            num_tasks (int): The number of tasks to create.
            seed (int): Seed value to ensure reproducibility of the split.
            split_type (str): The method to split the dataset into tasks. Supported types include:
                - "Naive": Divides the dataset evenly into the specified number of tasks.
                - "repeated_copies": Replicates the entire dataset for each task, useful for simulation studies.
                - "group_incremental": Split the dataset based on attribute group boundaries.
                - "class_incremental": Each task contains data from different classes, not dividing but isolating classes per task.
                - "group_ratios": Splits the dataset into tasks based on predefined ratios for each attribute group.
                - "group_probability": Randomly assigns data to tasks based on a probability distribution across groups.
                - "novel_disease": Specifically reserves one or more tasks to introduce novel diseases not seen in other tasks.
            split_ratios (dict, optional): A dictionary specifying the proportion of data to allocate to each task (used with "group_ratios").

        Returns:
            None: This method updates the dataset attributes directly, modifying train_set, val_set, and test_set's internal task structures.

        Raises:
            ValueError: If an unsupported `split_type` is provided.
            AssertionError: If `num_tasks` is less than required for the 'novel_disease' type.
        """
        self.num_tasks = num_tasks

        if split_type == "Naive":
            # Simply split the dataframe into the target number of tasks
            self.train_set.tasks_dataframe = np.array_split(self.train_set.dataframe, num_tasks)
            self.val_set.tasks_dataframe = np.array_split(self.val_set.dataframe, num_tasks)
            self.test_set.tasks_dataframe = np.array_split(self.test_set.dataframe, num_tasks)

        elif split_type == "repeated_copies":
            self.train_set.tasks_dataframe = [
                self.train_set.dataframe.copy() for _ in range(num_tasks)
            ]
            self.val_set.tasks_dataframe = [self.val_set.dataframe.copy() for _ in range(num_tasks)]
            self.test_set.tasks_dataframe = [
                self.test_set.dataframe.copy() for _ in range(num_tasks)
            ]

        elif split_type == "group_incremental":
            reference_attribute_groups = [
                group.attribute_group
                for group in self.train_set.get_attribute_groups("fine_grained_group")
            ]
            self.train_set.tasks_dataframe = self.split_group_incremental(
                self.train_set.get_attribute_groups("fine_grained_group"),
                reference_attribute_groups,
                self.name,
            )
            self.val_set.tasks_dataframe = self.split_group_incremental(
                self.val_set.get_attribute_groups("fine_grained_group"),
                reference_attribute_groups,
                self.name,
            )
            self.test_set.tasks_dataframe = self.split_group_incremental(
                self.test_set.get_attribute_groups("fine_grained_group"),
                reference_attribute_groups,
                self.name,
            )

        elif split_type == "class_incremental":
            warnings.warn("'num_tasks' ignored for 'class_incremental' split_type.")
            reference_targets = self.train_set.targets_set
            self.train_set.tasks_dataframe = self.split_class_incremental(
                self.train_set.dataframe, reference_targets
            )
            self.val_set.tasks_dataframe = self.split_class_incremental(
                self.val_set.dataframe, reference_targets
            )
            self.test_set.tasks_dataframe = self.split_class_incremental(
                self.test_set.dataframe, reference_targets
            )

        elif split_type == "group_ratios":
            (
                self.train_set.tasks_dataframe,
                modified_split_ratios,
            ) = self.split_by_groups_ratios(self.train_set, split_ratios, num_tasks, seed)
            self.val_set.tasks_dataframe, _ = self.split_by_groups_ratios(
                self.val_set, modified_split_ratios, num_tasks, seed, replace=True
            )
            self.test_set.tasks_dataframe, _ = self.split_by_groups_ratios(
                self.test_set, modified_split_ratios, num_tasks, seed, replace=True
            )

        elif split_type == "group_probability":
            attribute_groups = [group for group in self.train_set.get_attribute_groups()]
            if len(attribute_groups) == 1:
                # if only a single group exists (e.g., imbalanced client), revert back to a naive split
                self.train_set.tasks_dataframe = np.array_split(self.train_set.dataframe, num_tasks)
                self.val_set.tasks_dataframe = np.array_split(self.val_set.dataframe, num_tasks)
                self.test_set.tasks_dataframe = np.array_split(self.test_set.dataframe, num_tasks)
            else:
                (
                    self.train_set.tasks_dataframe,
                    percentage_values,
                    split_ratios,
                ) = self.split_by_group_probs(self.train_set, num_tasks, seed)
                self.val_set.tasks_dataframe, _ = self.split_by_groups_ratios(
                    self.val_set, split_ratios, num_tasks, seed, replace=True
                )
                self.test_set.tasks_dataframe, _ = self.split_by_groups_ratios(
                    self.test_set, split_ratios, num_tasks, seed, replace=True
                )

        elif split_type == "novel_disease":
            assert num_tasks >= 2
            self.train_set.tasks_dataframe = self.split_by_novel_disease(
                self.train_set, num_tasks, seed, self.novel_label
            )
            self.val_set.tasks_dataframe = np.array_split(self.val_set.dataframe, num_tasks)
            self.test_set.tasks_dataframe = np.array_split(self.test_set.dataframe, num_tasks)

        else:
            raise ValueError(f"Unsupported split_type: {split_type}.")

    @staticmethod
    def split_group_incremental(attribute_groups, reference_attribute_group, dataset_name):
        """
        Split a dataframe into tasks based on attribute groups.

        Args:
            attribute_groups (AttributeGroup list): List of attribute groups.
            reference_attribute_group (list): List of attribute groups to be used for sorting the split.
            dataset_name (int): Name of the dataset.

        Returns:
            list: List of dataframes, each representing a task.
        """
        # Sort the attribute_groups list based on the order of reference_attribute_group
        reference_attribute_group.sort()
        sorted_attribute_groups = []
        for attr_group in reference_attribute_group:
            # Iterate through each attribute group to find the matching object
            for ag in attribute_groups:
                if ag.attribute_group == attr_group:
                    sorted_attribute_groups.append(ag)
                    break

        dataframes_list = [group.group_df for group in sorted_attribute_groups]

        # For datasets with "Age_multi" attributes, combine age group 0 (0-20 years old) and age group 1 (20-40 years old).
        # This is done because most datasets have a limited number of samples in age group 0.
        if attribute_groups[0].attribute_names[0] == "Age_multi":
            # Check if groups 0 and 1 exist and merge them if so
            potential_group0 = sorted_attribute_groups[0]
            potential_group1 = sorted_attribute_groups[1]
            if (
                potential_group0.attribute_group[0] == 0
                and potential_group1.attribute_group[0] == 1
            ):
                concatenated_df = pd.concat(dataframes_list[:2])
                dataframes_list = [concatenated_df] + dataframes_list[2:]

        # df_sorted_by_attributes = pd.concat(dataframes_list)
        # np.array_split(df_sorted_by_attributes, num_tasks)
        return dataframes_list

    @staticmethod
    def split_class_incremental(dataframe, reference_targets):
        """
        Splits the dataframe into a list of dataframes based on unique targets.

        Parameters:
        - dataframe (pd.DataFrame): The input dataframe with a 'Target' column.
        - reference_targets (list): List of unique targets to split the dataframe.

        Returns:
        - list: A list of dataframes, each for a unique target.
        """
        tasks_list = []
        reference_targets = sorted(list(reference_targets))
        for target in reference_targets:
            # Filter the dataframe for the current target
            task_df = dataframe[dataframe["Target"] == target]
            tasks_list.append(task_df)
        return tasks_list

    @staticmethod
    def split_by_groups_ratios(dataset, split_ratios, num_tasks, seed, replace=False):
        """
        Create data splits based on attribute groups and split ratios.

        Args:
            dataset (BaseDataset): dataset to be split.
            split_ratios (list): List of lists representing split ratios for each task.
            num_tasks (int): Number of tasks.
            seed (int): Random seed for reproducibility.
            replace (boolean): Whether group sampling is done with replacement.

        Returns:
            List of dataframes, one for each task.
        """
        attribute_groups = [group for group in dataset.get_attribute_groups()]
        copied_attribute_groups = deepcopy(attribute_groups)

        # Find the attribute group with the smallest size
        smallest_group = min(copied_attribute_groups, key=lambda group: len(group))

        reserved_samples = pd.DataFrame()
        if not replace:
            # Adjust the sizes of larger attribute groups to match the smallest group
            for group in copied_attribute_groups:
                if group is smallest_group:
                    continue  # Skip the smallest group

                size_difference = len(group) - len(smallest_group)
                if size_difference > 0:
                    excluded_samples = group.sample(
                        size_difference
                    )  # Exclude additional samples to match the size of the smallest group
                    reserved_samples = pd.concat([reserved_samples, excluded_samples])
                elif size_difference < 0:
                    raise ValueError(
                        "Unexpected: A group is smaller in size to the smallest group."
                    )
        reserved_samples_size = len(reserved_samples)

        # Calculate the total number of available samples
        total_samples = sum(len(group.remaining_df) for group in copied_attribute_groups)
        samples_per_task = total_samples // num_tasks

        # Create task dataframes based on the predefined ratios
        dataframe_splits = []
        for i in range(num_tasks):
            task_df = pd.DataFrame()
            for j, ratio in enumerate(split_ratios[i]):
                num_group_samples = int(samples_per_task * ratio)
                sampled_df = copied_attribute_groups[j].sample(num_group_samples, replace)
                task_df = pd.concat([task_df, sampled_df])

            if reserved_samples_size > 0:
                num_reserved_samples_per_task = reserved_samples_size // num_tasks
                sampled_df = reserved_samples.sample(num_reserved_samples_per_task)
                task_df = pd.concat([task_df, sampled_df])
                reserved_samples = reserved_samples.drop(sampled_df.index)

            # Shuffle the task DataFrame randomly
            task_df = task_df.sample(frac=1.0, random_state=seed)
            dataframe_splits.append(task_df)

        # Adjust split ratios in response to reserved samples,
        # ensuring validation/test tasks mirror the training split proportions.
        adjusted_split_ratios = []
        if reserved_samples_size > 0:
            for df in dataframe_splits:
                temp_dataset = deepcopy(dataset).new_instance(df)
                counts = [len(group) for group in temp_dataset.get_attribute_groups()]
                ratio = (np.array(counts) / sum(counts)).tolist()
                adjusted_split_ratios.append(ratio)
        else:
            # If there are no reserved samples, use the original split ratios.
            adjusted_split_ratios = split_ratios

        return dataframe_splits, adjusted_split_ratios

    @staticmethod
    def split_by_novel_disease(dataset, num_tasks, seed, novel_label):
        """
        Create data splits based on a novel disease that appears in the second task.

        Args:
            dataset (BaseDataset): dataset to be split.
            num_tasks (int): Number of tasks.
            seed (int): Random seed for reproducibility.
            novel_label (tuple): Numeric label for the novel disease.

        Returns:
            List of dataframes, one for each task.
        """
        if num_tasks < 2:
            raise ValueError(
                "This splitting method is only supported for a sequence of two tasks or more."
            )

        # Get all the attribute groups from the dataset
        attribute_groups = [
            group
            for group in dataset.get_attribute_groups(attribute_type="target", is_sorted=False)
        ]

        # Find the Novel Disease Group and other groups
        novel_group = next(
            (group for group in attribute_groups if group.attribute_group == (novel_label,)),
            None,
        )
        other_groups = [
            group for group in attribute_groups if group.attribute_group != (novel_label,)
        ]

        # Total counts for novel and other groups
        total_novel_count = len(novel_group.remaining_df)
        total_other_count = sum(len(group.remaining_df) for group in other_groups)

        # Define the percentage of samples for each task for the novel disease and other groups
        if random.random() < 0.5:
            # Scenario 1: 0%, 0%, 10%, 50%
            novel_percentage_values = np.array([0, 0, 6, 35]) / 100.0
            # novel_percentage_values = np.array([0, 0, 12, 60])/100.0
            other_percentage_values = np.array([30, 30, 25, 15]) / 100.0
        else:
            # Scenario 2: 0%, 10%, 50%, >90%
            novel_percentage_values = np.array([0, 8, 27, 65]) / 100.0
            # other_percentage_values = np.array([77, 15, 7, 1])/100.0
            other_percentage_values = np.array([54, 30, 15, 1]) / 100.0

        # Calculate the total counts for novel and other groups
        total_novel_count = len(novel_group.remaining_df)
        total_other_counts = {
            group.attribute_group: len(group.remaining_df) for group in other_groups
        }

        # Pre-calculate the number of samples for each task for the novel group
        novel_counts_per_task = (novel_percentage_values * total_novel_count).astype(int)
        # novel_counts_per_task[-1] = total_novel_count - np.sum(novel_counts_per_task[:-1])  # Ensure all samples are allocated

        # Pre-calculate the number of samples for each task for other groups
        other_counts_per_task = {}
        for group in other_groups:
            group_counts = (
                other_percentage_values * total_other_counts[group.attribute_group]
            ).astype(int)
            group_counts[-1] = total_other_counts[group.attribute_group] - np.sum(
                group_counts[:-1]
            )  # Ensure all samples are allocated
            other_counts_per_task[group.attribute_group] = group_counts

        dataframe_splits = []
        for j in range(num_tasks):
            task_df = pd.DataFrame()

            # Sample from the novel group for the current task
            novel_samples_for_task = novel_group.sample(novel_counts_per_task[j])
            task_df = pd.concat([task_df, novel_samples_for_task])

            # Sample from each of the other groups for the current task
            for group in other_groups:
                group_samples_for_task = group.sample(
                    other_counts_per_task[group.attribute_group][j]
                )
                task_df = pd.concat([task_df, group_samples_for_task])

            # Shuffle the task's DataFrame
            task_df = task_df.sample(frac=1, random_state=seed)

            # Append the task DataFrame to the list
            dataframe_splits.append(task_df)

        return dataframe_splits

    @staticmethod
    def split_by_group_probs(dataset, num_tasks, seed, percentage_values=None):
        """
        Create data splits based on attribute groups using random sampling.

        Args:
            dataset (BaseDataset): dataset to be split.
            num_tasks (int): Number of tasks.
            seed (int): Random seed for reproducibility.
            percentage_values (list): Percentages of smallest group samples in each task.

        Returns:
            List of dataframes, one for each task.
            List of percentages of smallest group samples in each task.
            List of lists representing split ratios for each task.
        """
        attribute_groups = [group for group in dataset.get_attribute_groups()]
        if len(attribute_groups) != 2:
            raise ValueError("This splitting method is currently only supported for two groups.")

        copied_attribute_groups = deepcopy(attribute_groups)
        # np.random.seed(seed)  # Set the numpy random seed for reproducibility

        # Step 1: Compute Task Size
        total_samples = sum(len(group.remaining_df) for group in copied_attribute_groups)
        task_size = total_samples // num_tasks

        # Step 2: Find the Smallest Group
        smallest_group = min(copied_attribute_groups, key=lambda group: len(group))

        # Step 3: Sample Percentage Values
        if percentage_values is None:
            percentage_values = np.abs(np.random.normal(size=num_tasks))
            percentage_values /= percentage_values.sum()  # Normalize to ensure they sum up to 1.0

        num_samples_from_smallest_group = []
        for j in range(num_tasks):
            percentage = percentage_values[j]
            num_samples_from_smallest_group.append(
                min(int(percentage * len(smallest_group.remaining_df)), task_size)
            )

        # Step 4: Create Tasks
        dataframe_splits, split_ratios = [], []
        for j in range(num_tasks):
            task_df = pd.DataFrame()

            # Sample from the smallest group based on the percentage
            smallest_group_samples = smallest_group.sample(num_samples_from_smallest_group[j])
            task_df = pd.concat([task_df, smallest_group_samples])

            # Calculate the number of remaining samples needed
            remaining_samples_needed = task_size - len(task_df)

            # Randomly sample from the other group to fill the task
            # other_groups = [group for group in copied_attribute_groups if group is not smallest_group]
            task_ratios = [0] * len(copied_attribute_groups)
            for i, group in enumerate(copied_attribute_groups):
                if group is not smallest_group:
                    remaining_group_samples = len(group.remaining_df)
                    num_samples_to_sample = min(remaining_group_samples, remaining_samples_needed)
                    group_samples = group.sample(num_samples_to_sample)
                    task_df = pd.concat([task_df, group_samples])
                    task_ratios[i] = num_samples_to_sample
                else:
                    task_ratios[i] = num_samples_from_smallest_group[j]

            # Normalize the task ratios to make them sum up to 1
            task_samples = sum(task_ratios)
            task_ratios = [x / task_samples for x in task_ratios]

            # Shuffle the task's DataFrame
            task_df = task_df.sample(frac=1.0, random_state=seed)

            # Append the task DataFrame to the list
            dataframe_splits.append(task_df)
            split_ratios.append(task_ratios)

        return dataframe_splits, percentage_values, split_ratios

    def load_task(self, task_id, is_print=False):
        """
        Load a specific task for the DatasetSplit.

        Args:
            task_id (int): The ID of the task to be loaded.
            is_print (bool): Whether to print information about the loaded task (default: True).

        Returns:
            None. Updates the train_set, val_set, and test_set attributes by loading the specified task.

        """
        self.train_set.load_task(task_id, self.num_tasks, is_train=True, is_print=is_print)
        self.val_set.load_task(task_id, self.num_tasks, is_train=False, is_print=is_print)
        self.test_set.load_task(task_id, self.num_tasks, is_train=False, is_print=is_print)

    def __str__(self):
        """
        Returns a string representation of the DatasetSplit object.

        Returns:
            str: String representation of the DatasetSplit object.
        """
        return f"{self.name}(Train: {len(self.train_set)}, Val: {len(self.val_set)}, Test: {len(self.test_set)}, Num_classes: {self.num_classes}, ID: {self.client_id})"

    def __repr__(self):
        """
        Returns a string representation of the DatasetSplit object for debugging purposes.

        Returns:
            str: String representation of the DatasetSplit object.
        """
        return "\n" + str(self)

    def __lt__(self, other):
        """
        Compares two DatasetSplit objects based on the size of their training sets.

        Args:
            other (DatasetSplit): Another DatasetSplit object to compare against.

        Returns:
            bool: True if the training set of the current object is smaller than the other object, False otherwise.
        """
        return len(self.train_set) < len(other.train_set)
