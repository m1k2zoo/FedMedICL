from torch.utils.data import RandomSampler, Subset
from copy import deepcopy
import os
import json
import pandas as pd
import numpy as np
import random

from dataset.Dataset2D import Dataset2D
from dataset.CovidDataset2D import CovidDataset2D
from dataset.util.DatasetSplit import DatasetSplit
from reporting.visualize import (
    visualize_all_datasets,
    visualize_all_clients,
    visualize_all_clients_tasks,
)
from reporting.logging import store_all_clients_data, store_all_clients_tasks_data


def prepare_datasets(
    datasets,
    num_clients,
    seed,
    config_path,
    output_dir,
    task_split_type,
    is_imbalanced=False,
    imbalance_type="group",
    imbalance_ratios=None,
    is_novel_disease=False,
):
    """
    Load the dataset, distribute the dataset to clients and visualize the distributions.

    Args:
        datasets (list of str): List of names of the datasets to be loaded and distributed.
        num_clients (int): Number of clients to distribute the datasets across.
        seed (int): Seed value to ensure reproducibility.
        config_path (str): Path to the configuration JSON file that includes dataset paths and settings.
        output_dir (str): Directory where outputs like visualizations and client data will be stored.
        task_split_type (str): Defines the methodology to split datasets into tasks.
        is_imbalanced (bool): Flag to determine whether the distribution should consider imbalances (default: False).
        imbalance_type (str): Specifies the criterion ('group' or 'target') based on which imbalances are defined.
        imbalance_ratios (dict, optional): Specifies custom imbalanced ratios for clients, if not provided uses default from config.
        is_novel_disease (boolean): Whether some (but not all) clients have a novel disease target.
    Returns:
        datasets_list (list): Prepared datasets after loading and configuration.
        client_dataset_list (list of DatasetSplit): Datasets after distribution among clients.

    """
    with open(config_path, "r") as f:
        config = json.load(f)

    if output_dir is None:
        raise ValueError("output_dir should be provided.")
    visualization_dir = os.path.join(output_dir, "visualization")
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)

    # Load the dataset
    datasets_list = load_datasets(datasets, config, imbalance_type, is_novel_disease)

    # Shuffle data here since DataLoader shuffling is set to false.
    for dataset in datasets_list:
        dataset.train_set.shuffle_dataframe(seed)
        dataset.val_set.shuffle_dataframe(seed)
        dataset.test_set.shuffle_dataframe(seed)
        print(f"Sanity check for {dataset.name}, make sure the shuffling is the same for all runs")
        print(dataset.train_set.dataframe.head())

    # Visualize the attribute groups distribution of each dataset
    visualize_all_datasets(datasets_list, visualization_dir)

    if is_novel_disease:
        assert len(datasets_list) == 1  # Make sure only one dataset is passed
        novel_dataset = datasets_list[0]
        novel_dataset.novel_disease = True
        novel_dataset.novel_label = 2  # Assign COVID label as the novel label

    # Distribute the dataset to clients
    if is_imbalanced:
        if num_clients < 2:
            raise ValueError(
                "create_mixed_clients doesn't support experiments with less than 2 clients."
            )
        client_dataset_list = create_mixed_clients(
            datasets_list,
            num_clients,
            imbalance_type,
            imbalance_ratios,
            task_split_type,
            seed,
            config_path,
        )
    else:
        client_dataset_list = create_balanced_clients(datasets_list, num_clients)
    print(f"Client datasets are: {client_dataset_list}\n")

    # Visualize the attribute groups distribution of each client
    visualize_all_clients(datasets_list, client_dataset_list, visualization_dir)
    store_all_clients_data(client_dataset_list, data_dir)

    return datasets_list, client_dataset_list


def prepare_clients_tasks(
    datasets_list,
    client_dataset_list,
    num_tasks,
    seed,
    config_path,
    output_dir,
    task_split_type,
    custom_task_split_ratios,
    is_joint_training=False,
    is_novel_disease=False,
):
    """
    Create client tasks and visualize their distributions.

    Args:
        datasets_list (list): List of all datasets.
        client_dataset_list (list): List of client datasets.
        num_tasks (int): Number of tasks to create for each client dataset.
        seed (int): The seed value to reproduce the randomness.
        config_path (str): The path to the JSON configuration file.
        output_dir (str): The default output directory to save the visualization.
        task_split_type (str): Type of split to use for creating tasks.
                - "Naive": Divides the dataset evenly into the specified number of tasks.
                - "repeated_copies": Replicates the entire dataset for each task, useful for simulation studies.
                - "group_incremental": Split the dataset based on attribute group boundaries.
                - "class_incremental": Each task contains data from different classes, not dividing but isolating classes per task.
                - "group_ratios": Splits the dataset into tasks based on predefined ratios for each attribute group.
                - "group_probability": Randomly assigns data to tasks based on a probability distribution across groups.
                - "novel_disease": Specifically reserves one or more tasks to introduce novel diseases not seen in other tasks.
        custom_task_split_ratios (list): list of task split ratios (default: None).
        is_joint_training (bool): Flag indicating if the training is joint.
        is_novel_disease (boolean): Whether some (but not all) clients have a novel disease target.

    Returns:
        client_dataset_list (list): List of client datasets.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    if output_dir is None:
        raise ValueError("output_dir should be provided.")
    visualization_dir = os.path.join(output_dir, "visualization")
    data_dir = os.path.join(output_dir, "data")

    # Create tasks split for each client's dataset
    for client_dataset in client_dataset_list:
        task_split_ratios = None

        if is_novel_disease:
            if client_dataset.novel_disease:
                task_split_type = "novel_disease"
            else:
                task_split_type = "Naive"

        else:
            if task_split_type == "group_ratios":
                if custom_task_split_ratios:
                    print(
                        "Overriding default task split ratios with:",
                        custom_task_split_ratios,
                    )
                    task_split_ratios = custom_task_split_ratios
                else:
                    task_split_ratios = config[client_dataset.name]["task_split_ratios"]
                    print("Using default task split ratios:", task_split_ratios)

                num_attribute_groups = len(client_dataset.train_set.get_attribute_groups())

                assert (
                    len(task_split_ratios) == num_tasks
                ), "The length of task splits ratios should match the number of tasks."
                assert (
                    len(task_split_ratios[0]) == num_attribute_groups
                ), "The number of ratios per task should match the number of attribute groups."

        client_dataset.create_tasks(
            num_tasks, seed, split_type=task_split_type, split_ratios=task_split_ratios
        )

    # Visualize the attribute groups distribution of each task of each client
    if is_joint_training:
        visualization_dir = os.path.join(visualization_dir, "JT")
    visualize_all_clients_tasks(datasets_list, client_dataset_list, visualization_dir)
    store_all_clients_tasks_data(client_dataset_list, data_dir)

    return client_dataset_list


def load_datasets(dataset_names, config, imbalance_type, is_novel_disease):
    """
    Loads datasets based on provided names and configuration, adjusts for imbalances and novel disease settings.

    Args:
        dataset_names (list of str): Names of the datasets to load.
        config (dict): Configuration details for each dataset including paths and specific settings.
        imbalance_type (str): The type of imbalance to apply ('group' or 'target').
        is_novel_disease (bool): Whether some (but not all) clients have a novel disease target.

    Returns:
        list: List of DatasetSplit instances, each representing a loaded and configured dataset.
    """

    datasets_list = []
    for dataset_name in dataset_names:
        dataset_config = config[dataset_name]

        train_meta = pd.read_csv(dataset_config["train_meta_path"])
        val_meta = pd.read_csv(dataset_config["val_meta_path"])
        test_meta = pd.read_csv(dataset_config["test_meta_path"])
        dataset_type = dataset_config["dataset_type"]

        # If the dataset being processed is 'COVID', then also include the CheXpert dataset.
        # This is done by reading the CheXpert metadata for training, validation, and testing,
        # and then concatenating these with the corresponding COVID dataset metadata.
        if dataset_name == "COVID":
            # Only for covid
            train_meta = pd.concat([train_meta, val_meta])

            CheXpert_dataset_config = config["CheXpert"]
            CheXpert_train_meta = pd.read_csv(CheXpert_dataset_config["train_meta_path"])
            CheXpert_val_meta = pd.read_csv(CheXpert_dataset_config["val_meta_path"])
            CheXpert_test_meta = pd.read_csv(CheXpert_dataset_config["test_meta_path"])

            # Adjust the number of samples in CheXpert datasets to maintain a 12 to 1 ratio with COVID datasets

            # For training dataset
            COVID_len, CheXpert_len = len(train_meta), len(CheXpert_train_meta)
            samples_to_drop = CheXpert_train_meta.sample(CheXpert_len - 12 * COVID_len)
            CheXpert_train_meta = CheXpert_train_meta.drop(samples_to_drop.index)

            # For validation dataset
            COVID_len, CheXpert_len = len(val_meta), len(CheXpert_val_meta)
            samples_to_drop = CheXpert_val_meta.sample(CheXpert_len - 12 * COVID_len)
            CheXpert_val_meta = CheXpert_val_meta.drop(samples_to_drop.index)

            # For test dataset
            COVID_len, CheXpert_len = len(test_meta), len(CheXpert_test_meta)
            samples_to_drop = CheXpert_test_meta.sample(CheXpert_len - 12 * COVID_len)
            CheXpert_test_meta = CheXpert_test_meta.drop(samples_to_drop.index)

            if is_novel_disease:
                # Drop more samples from CheXpert for the novel experiment
                additional_samples_to_drop = CheXpert_train_meta.sample(10 * len(train_meta))
                CheXpert_train_meta = CheXpert_train_meta.drop(additional_samples_to_drop.index)
                additional_samples_to_drop = CheXpert_val_meta.sample(10 * len(val_meta))
                CheXpert_val_meta = CheXpert_val_meta.drop(additional_samples_to_drop.index)
                additional_samples_to_drop = CheXpert_test_meta.sample(10 * len(test_meta))
                CheXpert_test_meta = CheXpert_test_meta.drop(additional_samples_to_drop.index)

            train_meta = pd.concat([CheXpert_train_meta, train_meta])
            val_meta = pd.concat([CheXpert_val_meta, val_meta])
            test_meta = pd.concat([CheXpert_test_meta, test_meta])

        if dataset_type == "3D":
            # Placeholder for 3D dataset loading functionality.
            print("3D dataset functionality not implemented yet.")

        elif dataset_type == "2D":
            path_to_dataset = dataset_config["path_to_dataset"]
            sensitive_names = dataset_config["sensitive_names"]
            tasks_sensitive_name = dataset_config["tasks_sensitive_name"]

            # If the dataset is 'COVID', use the CovidDataset2D class with paths for both CheXpert and COVID datasets.
            # Otherwise, use the regular Dataset2D class with the single dataset path.
            if dataset_name == "COVID":
                CheXpert_path_to_dataset = CheXpert_dataset_config["path_to_dataset"]
                paths_to_dataset = {
                    "CheXpert": CheXpert_path_to_dataset,
                    "COVID": path_to_dataset,
                }
                train_set = CovidDataset2D(
                    train_meta,
                    paths_to_dataset,
                    sensitive_names,
                    tasks_sensitive_name,
                    None,
                )
                val_set = CovidDataset2D(
                    val_meta,
                    paths_to_dataset,
                    sensitive_names,
                    tasks_sensitive_name,
                    None,
                )
                test_set = CovidDataset2D(
                    test_meta,
                    paths_to_dataset,
                    sensitive_names,
                    tasks_sensitive_name,
                    None,
                )
            else:
                train_set = Dataset2D(
                    train_meta,
                    path_to_dataset,
                    sensitive_names,
                    tasks_sensitive_name,
                    None,
                )
                val_set = Dataset2D(
                    val_meta,
                    path_to_dataset,
                    sensitive_names,
                    tasks_sensitive_name,
                    None,
                )
                test_set = Dataset2D(
                    test_meta,
                    path_to_dataset,
                    sensitive_names,
                    tasks_sensitive_name,
                    None,
                )

        num_classes = len(train_set.dataframe["Target"].unique())
        dataset_split = DatasetSplit(
            dataset_name, train_set, val_set, test_set, num_classes, imbalance_type
        )
        datasets_list.append(dataset_split)

    return datasets_list


def generate_imbalance_scenarios(imbalance_ratios, num_clients):
    """
    Generate a distribution of clients among different imbalance scenarios based on specified ratios.

    Args:
    - imbalance_ratios (dict): A dictionary specifying the desired percentages of clients for each imbalance scenario.
      The keys are imbalance scenarios, and the values are the corresponding percentages.
    - num_clients (int): The total number of clients to distribute among the imbalance scenarios.

    Returns:
    - scenario_list (list): A list of imbalance scenarios.
    - num_spare_clients (int): The number of spare clients.
    """
    imbalance_scenarios = [
        "left_skewed",
        "right_skewed",
    ]  # , "left_missing", "right_missing"]

    if "spare" not in imbalance_ratios:
        raise ValueError("imbalance_ratios must include 'spare' scenario.")

    # Create a list with scenarios repeated according to specified ratios
    scenario_list = []
    num_spare_clients = 0
    for scenario, percentage in imbalance_ratios.items():
        if scenario == "spare":
            num_spare_clients = int(percentage * num_clients)
        else:
            scenario_list.extend([scenario] * max(int(percentage * num_clients), 1))

    # Fill in the remaining clients with other scenarios
    remaining_clients = num_clients - (len(scenario_list) + num_spare_clients)
    weighted_choices = [1 / len(imbalance_scenarios)] * len(imbalance_scenarios)

    while remaining_clients > 0:
        random_scenario = random.choices(imbalance_scenarios, weights=weighted_choices)[0]
        scenario_list.append(random_scenario)
        remaining_clients -= 1

        # Update the weighted_choices
        scenario_counts = {
            scenario: scenario_list.count(scenario) for scenario in imbalance_scenarios
        }
        weighted_choices = [
            1 / (0.1 + scenario_counts[scenario] ** 2) for scenario in imbalance_scenarios
        ]

    assert (
        len(scenario_list) + num_spare_clients == num_clients
    ), "The sum of scenario_list length and num_spare_clients must equal num_clients."

    return scenario_list, num_spare_clients


def create_balanced_clients(datasets_list, num_clients):
    """
    Distributes datasets over a given number of clients in a balanced manner.

    The method ensures each client receives roughly the same amount of data.
    If the number of datasets is less than the number of clients, the method
    takes the largest dataset and splits it into two equal halves. This process
    continues until the number of datasets matches the number of clients, ensuring
    a balanced distribution.

    Args:
        datasets_list (list): List of datasets (DatasetSplit) to be distributed.
        num_clients (int): Number of clients to distribute the datasets to.

    Returns:
        list: List of dataset splits distributed among clients in a balanced fashion.

    Precondition:
        The number of clients must be larger than or equal to the number of datasets.
    """
    # Check if the number of clients is greater than or equal to the number of datasets
    assert num_clients >= len(
        datasets_list
    ), "Number of clients must be larger than the number of datasets"

    datasets_list = deepcopy(datasets_list)
    # Sort the dataset list in descending order based on dataset size
    datasets_list.sort(reverse=True)

    # Distribute datasets until the number of clients is reached
    while len(datasets_list) < num_clients:
        # Take the largest dataset from the list and split it into two equal halves
        largest_split = datasets_list.pop(0)
        largest_split_clone = deepcopy(largest_split)
        dataframes_splits = np.array_split(largest_split.train_set.dataframe, 2)

        # Assign the splits to the original and cloned datasets
        largest_split.assign_train_set(dataframes_splits[0])
        largest_split_clone.assign_train_set(dataframes_splits[1])

        # Insert the dataset splits back into the list
        datasets_list.insert(0, largest_split)
        datasets_list.insert(0, largest_split_clone)

        # Sort the dataset list again in descending order based on dataset size
        datasets_list.sort(reverse=True)

    # Assign unique client IDs to each dataset split (each client)
    for i, dataset_split in enumerate(datasets_list):
        dataset_split.client_id = i
        dataset_split.train_set.client_id = i
        dataset_split.val_set.client_id = i
        dataset_split.test_set.client_id = i
        dataset_split.holdout_set.client_id = i

    # Return the list of dataset splits distributed among clients
    return datasets_list


def create_mixed_clients(
    datasets_list,
    num_clients,
    imbalance_type,
    custom_imbalance_ratios,
    task_split_type,
    seed,
    config_path,
):
    """
    Creates a mixture of imbalanced and balanced datasets for multiple clients. The largest dataset will be
    split over multiple clients with imbalanced data, while the remaining datasets will be distributed
    over clients in a balanced manner.

    Args:
        datasets_list (list): List of dataset objects to be processed.
        num_clients (int): The total number of clients the datasets should be split among.
        imbalance_type (str): Determine whether to create imbalanced clients based on groups or targets.
        custom_imbalance_ratios (dict): Custom definition of the imbalanced type for clients.
        task_split_type (str): Type of split to use for creating tasks.
        seed (int): The seed value to reproduce the randomness.
        config_path (str): Path to configuration containing imbalance types for the largest dataset.

    Returns:
        list: List of dataset objects split among clients, with the largest dataset having imbalanced data.

    Raises:
        AssertionError: If the number of clients is less than the number of datasets plus spare clients.
    """

    # Deepcopy ensures that the original dataset_list remains unchanged during processing.
    datasets_list = deepcopy(datasets_list)

    # Sort datasets in descending order based on their size.
    datasets_list.sort(reverse=True)

    if len(datasets_list) > 1:
        total_size = 0
        for dataset in datasets_list:
            total_size += len(dataset.train_set)

        size_per_client = total_size / num_clients

        num_clients_per_dataset = []
        for dataset in datasets_list:
            num_clients_per_dataset.append(round(len(dataset.train_set) / size_per_client))

        while sum(num_clients_per_dataset) > num_clients:
            num_clients_per_dataset[0] -= num_clients_per_dataset[0]

        while sum(num_clients_per_dataset) < num_clients:
            num_clients_per_dataset[-1] += num_clients_per_dataset[-1]

        imbalanced_clients_list = []
        for i, dataset in enumerate(datasets_list):
            sub_list = create_imbalanced_clients(
                dataset,
                num_clients_per_dataset[i],
                imbalance_type,
                custom_imbalance_ratios,
                task_split_type,
                seed,
                config_path,
            )
            imbalanced_clients_list.extend(sub_list)

        datasets_list = []

    else:
        # The largest dataset will be split with imbalances.
        largest_split = datasets_list.pop(0)
        imbalanced_clients_list = create_imbalanced_clients(
            largest_split,
            num_clients,
            imbalance_type,
            custom_imbalance_ratios,
            task_split_type,
            seed,
            config_path,
        )

    num_imbalanced_clients = len(imbalanced_clients_list)

    # Ensure that the total number of clients is larger than or equal to datasets + imbalanced splits.
    assert (
        num_clients >= len(datasets_list) + num_imbalanced_clients
    ), "Number of clients must be larger than the number of datasets"

    # Calculate the number of clients left after accounting for those with imbalanced data.
    remaining_clients = num_clients - num_imbalanced_clients

    # Continue splitting datasets until the number of datasets matches the remaining number of clients.
    while len(datasets_list) < remaining_clients:
        assert len(datasets_list) > 0, "Length of dataset list must be bigger than 0"

        # Split the largest dataset into two.
        largest_split = datasets_list.pop(0)
        largest_split_clone = deepcopy(largest_split)
        dataframes_splits = np.array_split(largest_split.train_set.dataframe, 2)

        # Assign the new data splits to the original and the cloned datasets.
        largest_split.assign_train_set(dataframes_splits[0])
        largest_split_clone.assign_train_set(dataframes_splits[1])

        # Add the split datasets back to the list.
        datasets_list.insert(0, largest_split)
        datasets_list.insert(0, largest_split_clone)

        # Re-sort the list based on size.
        datasets_list.sort(reverse=True)

    # Merge imbalanced splits with the other datasets.
    combined_dataset_list = imbalanced_clients_list + datasets_list

    # Assign unique client IDs to each dataset split.
    for i, dataset_split in enumerate(combined_dataset_list):
        dataset_split.client_id = i
        dataset_split.train_set.client_id = i
        dataset_split.val_set.client_id = i
        dataset_split.test_set.client_id = i
        dataset_split.holdout_set.client_id = i

    return combined_dataset_list


def reserve_excess_samples_from_dataset(dataset, dataset_attributes, seed):
    """
    Reserve excess samples from a dataset to prevent their allocation to a single client. This adjustment is
    crucial for maintaining a more balanced and realistic distribution of data across different clients.

    Args:
    - dataset (DatasetSplit): An instance of DatasetSplit containing the training dataset.
    - dataset_attributes (list): List of training group attributes.
    - seed: Random seed for reproducibility.

    Returns:
    - DataFrame of reserved samples.
    """

    # Find the largest and smallest attributes based on their lengths
    assert (
        len(dataset_attributes) == 2
    ), f"Expected only 2 attribute groups but found {len(dataset_attributes)} groups"
    largest_attribute = max(dataset_attributes, key=lambda attribute: len(attribute))
    smallest_attribute = min(dataset_attributes, key=lambda attribute: len(attribute))
    excess_samples = len(largest_attribute) - len(smallest_attribute)

    attribute_column = largest_attribute.attribute_names[0]
    attribute_value = largest_attribute.attribute_group[0]
    df = dataset.train_set.dataframe

    # Reserve excess samples
    reserved_samples = None
    if excess_samples > 0:
        attribute_df = df[df[attribute_column] == attribute_value]
        reserved_samples = attribute_df.sample(n=excess_samples, random_state=seed)
        df = df.drop(reserved_samples.index)

    dataset.assign_train_set(df)

    return reserved_samples


def distribute_samples_to_client(
    reserved_samples, training_dfs, client_index, num_samples_per_client
):
    """
    Distribute a specified number of reserved excess samples to a single client.

    Args:
    - reserved_samples (DataFrame): DataFrame containing the reserved excess samples.
    - training_dfs (list): List of DataFrames, each representing a client's training data.
    - client_index (int): Index of the client to receive the samples.
    - num_samples_per_client (int): Number of samples to distribute to the client.

    Returns:
    - training_dfs (list): Updated list of training DataFrames with distributed reserved samples for the specified client.
    - reserved_samples (DataFrame): Updated DataFrame of remaining reserved samples.
    """

    sampled_df = reserved_samples.sample(num_samples_per_client)
    training_dfs[client_index] = pd.concat(
        [training_dfs[client_index], sampled_df], ignore_index=True
    )
    reserved_samples = reserved_samples.drop(sampled_df.index)

    return training_dfs, reserved_samples


def create_imbalanced_clients(
    dataset,
    num_clients,
    imbalance_type,
    custom_imbalance_ratios,
    task_split_type,
    seed,
    config_path,
):
    """
    Splits a dataset into multiple client datasets with specified imbalances.

    The function includes a mechanism ('reserved_spare_ratio') to manage excess
    samples, ensuring they don't dominate in creating imbalances. These samples are
    evenly distributed among clients, preserving the desired imbalance
    distribution without any group being overrepresented.

    Args:
    - dataset (DatasetSplit): An instance of DatasetSplit to be split over multiple clients.
    - num_clients (int): The total number of clients the datasets should be split among.
    - imbalance_type (str): Determine whether to create imbalanced clients based on groups or targets.  Can be 'group' or 'target'
    - custom_imbalance_ratios (dict): Custom definition of the imbalanced type for clients. (default: None)
    - task_split_type (str): Type of split to use for creating tasks.
    - seed (int): Random seed for reproducibility.
    - config_path (str): Path to the JSON configuration file that specifies the type and number of imbalances.

    Returns:
    - new_dataset_list (list): List of datasets, each intended for a separate client, with the specified imbalances.

    """
    new_dataset_list = []
    reserved_spare_ratio = 0.7

    # Load configuration for imbalance types and their multiplicities
    with open(config_path, "r") as f:
        config = json.load(f)

    if custom_imbalance_ratios:
        print("Overriding default imbalance ratios with:", custom_imbalance_ratios)
        imbalance_ratios = custom_imbalance_ratios
    else:
        imbalance_ratios = config[dataset.name]["imbalance_ratios"]

    imbalance_scenario_list, num_spare_clients = generate_imbalance_scenarios(
        imbalance_ratios, num_clients
    )
    num_dataset_clients = len(imbalance_scenario_list) + num_spare_clients
    print(
        f"Creating the following imbalance scenarios: {imbalance_scenario_list} with {num_spare_clients} spare clients"
    )

    # Retrieve dataset attributes based on the selected attribute_type,
    # which can be either 'group' or 'target'.
    dataset_attributes = dataset.train_set.get_attribute_groups(attribute_type=imbalance_type)

    # Reserve excess samples to prevent their disproportionate allocation to any single client.
    reserved_samples = pd.DataFrame()
    if imbalance_type == "group":
        reserved_samples = reserve_excess_samples_from_dataset(dataset, dataset_attributes, seed)
    reserved_samples_size = len(reserved_samples)

    dataset_attributes = dataset.train_set.get_attribute_groups(attribute_type=imbalance_type)

    training_dfs = [pd.DataFrame()] * num_dataset_clients
    samples_per_client = len(dataset.train_set) / num_dataset_clients

    # ======= Create imbalanced clients =======
    client_index = 0
    for imbalance_scenario in imbalance_scenario_list:
        # This function needs to return a list of counts for each attribute group based on the imbalance ratio
        group_counts = calculate_attribute_distribution(
            imbalance_scenario, dataset_attributes, samples_per_client
        )
        for group_index, attribute_group in enumerate(dataset_attributes):
            # Ensure we don't sample more data than available
            if group_counts[group_index] > len(attribute_group.remaining_df):
                print(
                    f"Reducing required samples for the imbalance scenario '{imbalance_scenario}' due to insufficient available samples:"
                )
                group_counts[group_index] = len(attribute_group.remaining_df)
                # raise ValueError(f"Attempted to sample {group_counts[group_index]} samples from an attribute group of size {len(attribute_group)}. Please ensure the specified imbalances and client configurations do not exhaust available samples.")

            sampled_df = attribute_group.sample(group_counts[group_index])
            training_dfs[client_index] = pd.concat(
                [training_dfs[client_index], sampled_df], ignore_index=True
            )

        # Evenly distribute a ratio of reserved excess samples among clients.
        num_reserved_samples_per_client = int(
            (1 - reserved_spare_ratio) * reserved_samples_size
        ) // len(imbalance_scenario_list)
        distribute_samples_to_client(
            reserved_samples,
            training_dfs,
            client_index,
            num_reserved_samples_per_client,
        )

        client_index += 1

    # ======= Create spare clients =======
    # Compute group counts for the num_spare_clients default clients
    if num_spare_clients > 0:
        group_counts = []
        for j, attribute_group in enumerate(dataset_attributes):
            group_size = len(attribute_group.remaining_df)
            group_counts.append(group_size // num_spare_clients)

        # Fill the data for the num_spare_clients default clients
        current_index = client_index
        for client_index in range(current_index, num_dataset_clients):
            for group_index, attribute_group in enumerate(dataset_attributes):
                sampled_df = attribute_group.sample(group_counts[group_index])
                training_dfs[client_index] = pd.concat(
                    [training_dfs[client_index], sampled_df], ignore_index=True
                )

            # Evenly distribute a ratio of reserved excess samples among spare clients.
            num_reserved_samples_per_client = int(
                reserved_spare_ratio * reserved_samples_size
            ) // len(imbalance_scenario_list)
            distribute_samples_to_client(
                reserved_samples,
                training_dfs,
                client_index,
                num_reserved_samples_per_client,
            )

    # Clone the dataset for each client and shuffle the DataFrame for uniform attribute distribution
    for j in range(num_dataset_clients):
        cloned_client_dataset = deepcopy(dataset)
        training_dfs[j] = training_dfs[j].sample(frac=1, random_state=seed)
        cloned_client_dataset.assign_train_set(training_dfs[j])
        new_dataset_list.append(cloned_client_dataset)

    for client_index, imbalance_scenario in enumerate(imbalance_scenario_list):
        if imbalance_scenario == "right_missing":
            new_dataset_list[client_index].novel_disease = False
            # for dataset in new_dataset_list: print(dataset.novel_disease)

    # Return the list of dataset splits distributed among clients
    return new_dataset_list


def calculate_attribute_distribution(imbalance_scenario, dataset_attributes, samples_per_client):
    """
    Calculate the distribution of samples for attribute based on the specified imbalance scenario.

    Args:
    - imbalance_scenario (str): Specifies the scenario of imbalance of ('balanced', 'left_skewed', 'right_skewed', 'left_missing', 'right_missing').
    - dataset_attributes (list): List of training group attributes
    - samples_per_client (int): Total number of samples that should be distributed among the attribute.

    Returns:
    - group_counts (list of int): List of counts specifying the number of samples each attribute  should receive.
    """
    ratios = []
    num_attributes = len(dataset_attributes)

    if imbalance_scenario == "balanced":
        # Each group gets an equal share
        ratios = [1.0 / num_attributes] * num_attributes

    elif imbalance_scenario == "left_skewed":
        # First group gets 0.8, the rest share the remaining 0.2
        ratios = [0.8] + [(0.2 / (num_attributes - 1))] * (num_attributes - 1)

    elif imbalance_scenario == "right_skewed":
        # Last group gets 0.8, the rest share the remaining 0.2
        ratios = [(0.2 / (num_attributes - 1))] * (num_attributes - 1) + [0.8]

    elif imbalance_scenario == "left_missing":
        # First group gets nothing, the rest follows the dataset distribution
        attributes = deepcopy(dataset_attributes)
        attributes.pop(0)
        counts = [len(attribute) for attribute in attributes]
        ratios = [0.0] + (np.array(counts) / sum(counts)).tolist()

    elif imbalance_scenario == "right_missing":
        # Last group gets nothing, the rest follows the dataset distribution
        attributes = deepcopy(dataset_attributes)
        attributes.pop(-1)
        counts = [len(attribute) for attribute in attributes]
        ratios = (np.array(counts) / sum(counts)).tolist() + [0.0]

    else:
        raise ValueError(f"Unsupported imbalance_ratio {imbalance_scenario}")

    # Convert ratios to counts
    group_counts = [int(ratio * samples_per_client) for ratio in ratios]

    return group_counts
