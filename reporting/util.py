import traceback
import os
import warnings
import pandas as pd
import numpy as np
import pickle
import ast
from reporting.metrics import compute_macro_accuracy


def read_pickle_file(output_dir, file_name):
    """
    Reads pickle results from a directory.

    Args:
        output_dir (str): The directory containing pickle files.
        file_name (str): The pickle file name

    Returns:
        Any: The data loaded from the specified pickle file.
    """

    pickle_file_path = os.path.join(output_dir, file_name)

    # Load data from the pickle file
    with open(pickle_file_path, "rb") as file:
        data = pickle.load(file)

    return data


def read_csv_results(log_dir, file_suffix):
    """
    Reads CSV results from a directory, returning the dataframe of the specified model type.

    Args:
        log_dir (str): The directory containing CSV files.
        file_suffix (str): The expected suffix for the file names (e.g., "training").

    Returns:
        pd.DataFrame: A dataframe containing the specified model type's results.
    """

    # Initialize an empty dataframe for the specified model results
    model_df = None

    # List all CSV files in the directory and sort them
    csv_files = sorted(
        [f for f in os.listdir(log_dir) if f.endswith(".csv") and f[:-4].endswith(file_suffix)]
    )

    # Iterate through the sorted CSV files
    for filename in csv_files:
        df = pd.read_csv(os.path.join(log_dir, filename))

        # Concatenate dataframes from all clients
        if model_df is None:
            model_df = df
        else:
            model_df = pd.concat([model_df, df])

    return model_df


def task_matrix_from_trackers(tracker_data, client_id, split, metric_name):
    """
    Generates a matrix of per-category accuracy between tasks from tracker data.

    Args:
        tracker_data (dict): Dictionary with tracker objects keyed by client IDs.
        client_id (int): The client ID.
        split (str): The split to computed the accuracy on (e.g., "test")

    Returns:
        pd.DataFrame: A square DataFrame where indices and columns represent training and evaluation tasks

    """
    index_attribute = "train_task_id"
    column_attribute = "eval_task_id"

    num_tasks = tracker_data[client_id].num_tasks
    acc_matrix = pd.DataFrame(
        np.zeros((num_tasks, num_tasks)), columns=range(num_tasks), index=range(num_tasks)
    )

    for i in range(num_tasks):
        for j in range(num_tasks):
            acc, per_category_acc_dict, _ = tracker_data[client_id].average_acc(split, i, j)
            auc, per_category_auc_dict = tracker_data[client_id].get_auc(split, i, j)

            if metric_name == f"{split}_per_category_acc":
                per_category_acc = compute_macro_accuracy(per_category_acc_dict)
                acc_matrix.iloc[i, j] = per_category_acc
            elif metric_name == f"{split}_acc":
                acc_matrix.iloc[i, j] = acc
            elif metric_name == f"{split}_auc":
                acc_matrix.iloc[i, j] = auc

    acc_matrix.index.name = "train_task_id"
    acc_matrix.columns.name = "eval_task_id"
    return acc_matrix


def aggregate_metric_over_clients(df, metric_column):
    # Group the DataFrame by 'iteration' and calculate the mean, minimum and maximum for the specified metric
    agg_df = df.groupby("iteration")[metric_column].agg([np.mean, np.min, np.max]).reset_index()
    # Extract x and y values

    x = agg_df["iteration"]
    mean_values = agg_df["mean"]
    min_values = agg_df["amin"]
    max_values = agg_df["amax"]

    return x, mean_values, min_values, max_values


def construct_matrix_from_dataframe(metric_column, df):
    """
    Constructs an evaluation matrix from the given dataframe

    Args:
    - metric_column (str): The column name in the dataframe containing the metrics for the heatmap.
    - df (pd.DataFrame): Input dataframe containing the data.

    Returns:
    - matrix (pd.DataFrame): The constructed matrix
    """
    index_attribute = "train_task_id"
    column_attribute = "eval_task_id"

    if df is None:
        return None

    tasks_results_df = pd.DataFrame()
    unique_train_task_ids = df[index_attribute].unique()
    unique_eval_task_ids = df[column_attribute].unique()

    for train_id in unique_train_task_ids:
        for eval_id in unique_eval_task_ids:
            subset_df = df[(df[index_attribute] == train_id) & (df[column_attribute] == eval_id)]

            # Get results of the last iteration in this task combination
            last_iteration = subset_df["iteration"].max()
            task_result = subset_df[subset_df["iteration"] == last_iteration]

            if "client_id" in task_result.columns:
                # Aggregate over clients
                aggregated = task_result.groupby("iteration")[metric_column].mean().reset_index()
                aggregated[index_attribute] = train_id
                aggregated[column_attribute] = eval_id
                task_result = aggregated

            if len(task_result) > 1:
                # If this warning is triggered, something is wrong
                # We shouldn't have more than one row for the last iteration
                # Inspect the log files
                warnings.warn(
                    f"Multiple rows found for the same 'last iteration' in the subset for train_task_id {train_id} and eval_task_id {eval_id}."
                )
            tasks_results_df = tasks_results_df.append(task_result)

    matrix = tasks_results_df.pivot(
        index=index_attribute, columns=column_attribute, values=metric_column
    )

    # Assert that the number of rows equals the number of columns
    rows, columns = matrix.shape
    assert (
        rows == columns
    ), "Expected the number of rows to equal the number of columns in construct_matrix_from_dataframe()."

    return matrix


def compute_cumulative_task_accuracy(matrix_df):
    """
    Computes the cumulative accuracy over a series of tasks.

    Args:
    - matrix_df (DataFrame): A square DataFrame where matrix[i, j] represents the accuracy
      of the model trained up to task i when evaluated on task j.

    Returns:
    - cumulative_acc (list): A list containing the cumulative average of accuracies
      after each training task, computed over the test tasks seen up to that point.
    """
    cumulative_averages = []
    all_elements = []

    for i in range(len(matrix_df)):
        current_row = matrix_df.iloc[i, : i + 1]
        all_elements.extend(current_row.values.tolist())
        cumulative_average = np.mean(all_elements)
        cumulative_averages.append(cumulative_average)

    return cumulative_averages


def compute_all_clients_model_accuracy(exp_path, metric_name):
    """
    Computes accuracies for all clients for a specified model within an experiment.

    Parameters:
    - exp_path (str): Directory path to the experiment.
    - metric_name (str): Name of the metric to analyze (e.g., 'val_acc', 'holdout_acc').

    Returns:
    - list: A list containing the computed accuracies of the specified model for each client.
    """
    try:
        pickle_filename = "client_trackers_exp.pkl"
        file_path = os.path.join(exp_path, pickle_filename)
        if not os.path.exists(file_path):
            print(f"Results pickle file does not exist: {file_path}.")
            return None

        trackers_data = read_pickle_file(exp_path, pickle_filename)
        if "holdout" in metric_name:
            all_clients_holdout_accuracies = []
            for client_id in range(len(trackers_data)):
                if "acc" in metric_name:
                    _, holdout_per_category_acc_dict, _ = trackers_data[client_id].average_acc(
                        "holdout", 0
                    )
                    holdout_per_category_acc = compute_macro_accuracy(holdout_per_category_acc_dict)
                    all_clients_holdout_accuracies.append(holdout_per_category_acc)
                elif "auc" in metric_name:
                    holdout_auc, _ = trackers_data[client_id].get_auc("holdout", 0)
                    all_clients_holdout_accuracies.append(holdout_auc)

            # results = read_csv_results(exp_path, "holdout")
            # all_clients_holdout_accuracies = results[metric_name].values
            return all_clients_holdout_accuracies

        else:
            all_clients_task_accuracies = []

            split_name = ""
            if "val" in metric_name:
                split_name = "val"
            elif "test" in metric_name:
                split_name = "test"

            number_of_clients = len(trackers_data)
            for client_id in list(range(number_of_clients)):
                per_category_acc_matrix = task_matrix_from_trackers(
                    trackers_data, client_id, split_name, metric_name
                )
                client_task_accuracies = compute_cumulative_task_accuracy(per_category_acc_matrix)
                all_clients_task_accuracies.append(client_task_accuracies)
            return all_clients_task_accuracies
    except Exception as e:
        print(f"An error occurred while parsing an experiment results: {e}")
        traceback.print_exc()
        return 0


def compute_novel_disease_results(exp_path, algorithm, metric_name):
    """
    Computes disease-specific and aggregated metrics for a specific model type within an experiment.
    This function analyzes the performance of a model on common and novel diseases by computing
    the average accuracy for different groups of clients based on their labels.

    Parameters:
    - exp_path (str): The directory path where the experiment results are stored.
    - algorithm (str): The name of the algorithm being analyzed.
    - metric_name (str): The name of the metric to analyze, such as 'val_acc' or 'holdout_acc'.

    """
    all_clients_task_accuracies = []
    df = read_csv_results(exp_path, f"{algorithm}_holdout")
    test_df = read_csv_results(exp_path, f"{algorithm}_test")

    # Number of labels
    metric_dict = df[metric_name].values[0]
    metric_dict = ast.literal_eval(metric_dict)
    number_of_labels = len(metric_dict.keys())
    if number_of_labels != 3:
        raise ValueError(f"Expected three labels in the results but found {number_of_labels}")

    # Number of clients
    number_of_clients = df.client_id.max() + 1
    number_of_tasks = test_df.train_task_id.max() + 1
    if len(df) != number_of_tasks * number_of_clients:
        raise ValueError(
            f"Expected number of rows to be {number_of_tasks} times the number of clients but found {len(df)}"
        )

    # Iterations:
    iterations = df.iteration.unique()

    common_accuracies_mean, common_accuracies_std = [], []
    novel_accuracies_mean, novel_accuracies_std = [], []
    for iteration in iterations:
        df_iteration = df[df.iteration == iteration]
        clients_common_acc = []
        clients_novel_acc = []
        for client_id in range(number_of_clients):
            df_client = df_iteration[df_iteration.client_id == client_id]
            metric_dict = ast.literal_eval(df_client[metric_name].values[0])

            common_acc = (metric_dict[0] + metric_dict[1]) / 2
            novel_acc = metric_dict[2]

            clients_common_acc.append(common_acc)
            clients_novel_acc.append(novel_acc)

        # Calculating mean and standard deviation

        common_mean = np.array(clients_common_acc).mean()
        common_std = np.array(clients_common_acc).std()
        common_accuracies_mean.append(common_mean)
        common_accuracies_std.append(common_std)

        novel_mean = np.array(clients_novel_acc).mean()
        novel_std = np.array(clients_novel_acc).std()
        novel_accuracies_mean.append(novel_mean)
        novel_accuracies_std.append(novel_std)

    return (
        common_accuracies_mean,
        common_accuracies_std,
        novel_accuracies_mean,
        novel_accuracies_std,
    )
