import os
import csv
import pandas as pd
import warnings
import numpy as np
from reporting.metrics import compute_macro_accuracy
from reporting.util import read_csv_results, aggregate_metric_over_clients


def get_split_set(dataset, split):
    """
    Returns the specified split set from a given dataset.

    Parameters:
    - dataset (DatasetSplit object): The dataset from which to retrieve the split set.
    - split (str): The split set specified by the split parameter. Expect 'train', 'val', 'test or 'holdout'
    """
    if split == "train":
        return dataset.train_set
    elif split == "val":
        return dataset.val_set
    elif split == "test":
        return dataset.test_set
    elif split == "holdout":
        return dataset.holdout_set
    else:
        raise ValueError("Invalid split. Expected 'train', 'val', 'test' or 'holdout'.")


def store_all_clients_data(client_dataset_list, parent_dir):
    """
    Saves each client in separate CSV files.

    Parameters:
    - client_dataset_list (list of DatasetSplit objects): A list of client splits.
    - parent_dir (str): The parent directory to save the CSV files in.
    """
    clients_data_dir = os.path.join(parent_dir, "clients")
    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(clients_data_dir, exist_ok=True)

    # Store tasks for each client split
    for client_dataset in client_dataset_list:
        for split_name in ["train", "val", "test", "holdout"]:
            csv_filename = os.path.join(
                clients_data_dir,
                f"{split_name}_client{client_dataset.client_id}_{client_dataset.name}.csv",
            )
            get_split_set(client_dataset, split_name).dataframe.to_csv(csv_filename, index=False)


def store_all_clients_tasks_data(client_dataset_list, parent_dir):
    """
    Saves each client's tasks in separate CSV files.

    Parameters:
    - client_dataset_list (list of DatasetSplit objects): A list of client splits.
    - parent_dir (str): The parent directory to save the CSV files in.
    """
    tasks_data_dir = os.path.join(parent_dir, "tasks")
    os.makedirs(parent_dir, exist_ok=True)
    os.makedirs(tasks_data_dir, exist_ok=True)

    # Store tasks for each client split
    for client_dataset in client_dataset_list:
        for task_id in range(client_dataset.num_tasks):
            client_dataset.load_task(task_id, is_print=False)
            for split_name in ["train", "val", "test"]:
                csv_filename = os.path.join(
                    tasks_data_dir,
                    f"{split_name}_client{client_dataset.client_id}_task{task_id}_{client_dataset.name}.csv",
                )
                get_split_set(client_dataset, split_name).dataframe.to_csv(
                    csv_filename, index=False
                )


def log_summary_metrics_to_wandb(log_dir, split_name, algorithm, wandb=None):
    """
    Generate and report summary metrics for a given split.

    Args:
        log_dir (str): The directory containing log files.
        split_name (str): The name of the split, e.g., "test", or "val"
        algorithm (str): The name of the algorithm
        wandb (object, optional): An optional Weights and Biases (wandb) object for reporting.
            If not provided, the reporting to wandb will be skipped.

    Returns:
        None
    """
    if not wandb:
        return

    # Read CSV results
    df = read_csv_results(log_dir, split_name)

    # Aggregate metrics
    if df is None:
        warnings.warn(
            f"Skipping reporting summary metrics for {split_name} of {algorithm} as dataframe results could not be found."
        )
    else:
        _, mean_acc, min_acc, max_acc = aggregate_metric_over_clients(df, f"{split_name}_acc")
        _, mean_category_acc, min_category_acc, max_category_acc = aggregate_metric_over_clients(
            df, f"{split_name}_per_category_acc"
        )
        _, mean_group_acc, min_group_acc, max_group_acc = aggregate_metric_over_clients(
            df, f"{split_name}_per_group_acc"
        )

        # Report summary results
        wandb.run.summary[f"{split_name}/mean_accuracy/{algorithm}"] = mean_acc.values[-1]
        wandb.run.summary[f"{split_name}/min_accuracy/{algorithm}"] = min_acc.values[-1]
        wandb.run.summary[f"{split_name}/max_accuracy/{algorithm}"] = max_acc.values[-1]

        wandb.run.summary[f"{split_name}/mean_per_category_accuracy/{algorithm}"] = (
            mean_category_acc.values[-1]
        )
        wandb.run.summary[f"{split_name}/min_per_category_accuracy/{algorithm}"] = (
            min_category_acc.values[-1]
        )
        wandb.run.summary[f"{split_name}/max_per_category_accuracy/{algorithm}"] = (
            max_category_acc.values[-1]
        )

        wandb.run.summary[f"{split_name}/mean_per_group_accuracy/{algorithm}"] = (
            mean_group_acc.values[-1]
        )
        wandb.run.summary[f"{split_name}/min_per_group_accuracy/{algorithm}"] = (
            min_group_acc.values[-1]
        )
        wandb.run.summary[f"{split_name}/max_per_group_accuracy/{algorithm}"] = (
            max_group_acc.values[-1]
        )


def append_data_to_csv(filename, output_dir, data, header):
    """
    Appends data to a CSV file.

    Args:
        filename (str): The name of the CSV file to which data will be appended.
        output_dir (str): The parent directory for the experiment outputs.
        data (list): The data to be appended to the file.
        header (list): The header to be added to the file if it doesn't exist.

    Returns:
        None

    Notes:
        If the CSV file specified by 'filename' does not exist, the 'header' will
        be written as the first row in the file. Subsequent calls will append
        'data' to the existing file.
    """
    log_dir = os.path.join(output_dir, "logs")  # Create a 'logs' subdirectory
    os.makedirs(log_dir, exist_ok=True)  # Ensure the 'logs' directory exists
    log_file_path = os.path.join(log_dir, filename)

    header_exists = os.path.exists(log_file_path)

    with open(log_file_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not header_exists:
            writer.writerow(header)  # Add the header if it doesn't exist

        writer.writerow(data)


def log_training_results(metrics_tracker, task_id, iteration, lr, wandb, output_dir):
    """
    Logs the losses and accuracies of the current iteration using WandB and optionally to a text file.

    Args:
        metrics_tracker (ClientTrackers): An object that tracks metrics for a specific client.
        task_id (int): The task ID.
        iteration (int): The current iteration.
        lr (float): The current learning rate.
        wandb: WandB object for logging.
        output_dir (str): The parent directory for the experiment outputs.
    """

    client_id = metrics_tracker.client_id

    algorithm = metrics_tracker.algorithm
    filename = f"{algorithm}_{client_id}_training.csv"

    sub_exp_id = algorithm
    if algorithm != "JT":
        sub_exp_id = f"{algorithm}_{client_id}"

    train_loss = metrics_tracker.average_loss("train", task_id)
    train_acc, train_per_category_acc_dict, train_per_group_acc_dict = metrics_tracker.average_acc(
        "train", task_id
    )
    train_per_category_acc = compute_macro_accuracy(train_per_category_acc_dict)
    train_per_group_acc = compute_macro_accuracy(train_per_group_acc_dict)

    train_auc, train_per_category_auc_dict = metrics_tracker.get_auc("train", task_id)

    data = [
        iteration,
        client_id,
        task_id,
        train_loss,
        train_acc,
        train_per_category_acc,
        train_per_category_acc_dict,
        train_per_group_acc,
        train_per_group_acc_dict,
        train_auc,
        train_per_category_auc_dict,
    ]
    header = [
        "iteration",
        "client_id",
        "task_id",
        "train_loss",
        "train_acc",
        "train_per_category_acc",
        "train_per_category_acc_dict",
        "train_per_group_acc",
        "train_per_group_acc_dict",
        "train_auc",
        "train_per_category_auc_dict",
    ]

    # Call the function to append the data to the CSV file
    append_data_to_csv(filename, output_dir, data, header)

    if wandb:
        wandb_log_data = {
            f"train/iteration": iteration,
            f"train/task_id/{sub_exp_id}": task_id,
            f"train/lr/{sub_exp_id}": lr,
            f"train/loss/{sub_exp_id}": train_loss,
            f"train/accuracy/{sub_exp_id}": train_acc,
            f"train/per_category_accuracy/{sub_exp_id}": train_per_category_acc,
            f"train/per_group_accuracy/{sub_exp_id}": train_per_group_acc,
        }

        wandb.log(wandb_log_data)


def log_evaluation_results(
    split, metrics_tracker, train_task_id, eval_task_id, iteration, wandb, output_dir
):
    """
    Logs the losses and accuracies of the current iteration to a text file.

    Args:
        metrics_tracker (ClientTrackers): An object that tracks metrics for a specific client.
        train_task_id (int): The training task ID.
        eval_task_id (int): The evaluation task ID.
        iteration (int): The current iteration.
        wandb: WandB object for logging.
        output_dir (str): The parent directory for the experiment outputs.
    """
    client_id = metrics_tracker.client_id

    loss = metrics_tracker.average_loss(split, train_task_id, eval_task_id)
    acc, per_category_acc_dict, per_group_acc_dict = metrics_tracker.average_acc(
        split, train_task_id, eval_task_id
    )
    per_category_acc = compute_macro_accuracy(per_category_acc_dict)
    per_group_acc = compute_macro_accuracy(per_group_acc_dict)

    auc, per_category_auc_dict = metrics_tracker.get_auc(split, train_task_id, eval_task_id)

    data = [
        iteration,
        train_task_id,
        eval_task_id,
        loss,
        acc,
        per_category_acc,
        per_category_acc_dict,
        per_group_acc,
        per_group_acc_dict,
        auc,
        per_category_auc_dict,
    ]
    header = [
        "iteration",
        "train_task_id",
        "eval_task_id",
        f"{split}_loss",
        f"{split}_acc",
        f"{split}_per_category_acc",
        f"{split}_per_category_acc_dict",
        f"{split}_per_group_acc",
        f"{split}_per_group_acc_dict",
        f"{split}_auc",
        f"{split}_per_category_auc_dict",
    ]

    algorithm = metrics_tracker.algorithm
    filename = f"{algorithm}_{split}.csv"
    sub_exp_id = "JT"
    if algorithm != "JT":
        sub_exp_id = f"{algorithm}_{client_id}"

    if client_id != -1:
        data.insert(1, client_id)
        header.insert(1, "client_id")

    # Call the function to append the data to the CSV file
    append_data_to_csv(filename, output_dir, data, header)

    if wandb and eval_task_id == train_task_id:
        loss_over_seen_tasks = metrics_tracker.average_loss_over_seen_tasks(split, train_task_id)
        wandb_log_data = {
            f"{split}/iteration": iteration,
            f"{split}/task_id/{sub_exp_id}": train_task_id,
            f"{split}/loss/{sub_exp_id}": loss_over_seen_tasks,
            f"{split}/accuracy/{sub_exp_id}": acc,
            f"{split}/per_category_accuracy/{sub_exp_id}": per_category_acc,
            f"{split}/per_group_accuracy/{sub_exp_id}": per_group_acc,
        }
        wandb.log(wandb_log_data)


def log_holdout_results(metrics_tracker, iteration, wandb, output_dir):
    """
    Logs the losses and accuracies of the holdout to a text file.

    Args:
        metrics_tracker (ClientTrackers): An object that tracks metrics for a specific client.
        iteration (int): The current iteration.
        wandb: WandB object for logging.
        output_dir (str): The parent directory for the experiment outputs.
    """
    client_id = metrics_tracker.client_id
    algorithm = metrics_tracker.algorithm

    filename = f"{algorithm}_holdout.csv"
    sub_exp_id = algorithm
    if algorithm != "JT":
        sub_exp_id = f"{algorithm}_{client_id}"

    task_id = 0  # holdout tracker only keep tracks of a single task (last task)
    holdout_loss = metrics_tracker.average_loss("holdout", task_id)
    (
        holdout_acc,
        holdout_per_category_acc_dict,
        holdout_per_group_acc_dict,
    ) = metrics_tracker.average_acc("holdout", task_id)
    holdout_per_category_acc = compute_macro_accuracy(holdout_per_category_acc_dict)
    holdout_per_group_acc = compute_macro_accuracy(holdout_per_group_acc_dict)

    holdout_auc, holdout_per_category_auc_dict = metrics_tracker.get_auc("holdout", task_id)

    data = [
        iteration,
        holdout_loss,
        holdout_acc,
        holdout_per_category_acc,
        holdout_per_category_acc_dict,
        holdout_per_group_acc,
        holdout_per_group_acc_dict,
        holdout_auc,
        holdout_per_category_auc_dict,
    ]
    header = [
        "iteration",
        "holdout_loss",
        "holdout_acc",
        "holdout_per_category_acc",
        "holdout_per_category_acc_dict",
        "holdout_per_group_acc",
        "holdout_per_group_acc_dict",
        "holdout_auc",
        "holdout_per_category_auc_dict",
    ]

    if client_id != -1:
        data.insert(1, client_id)
        header.insert(1, "client_id")

    # Call the function to append the data to the CSV file
    append_data_to_csv(filename, output_dir, data, header)

    if wandb:
        wandb_log_data = {
            f"holdout/iteration": iteration,
            f"holdout/loss/{sub_exp_id}": holdout_loss,
            f"holdout/accuracy/{sub_exp_id}": holdout_acc,
            f"holdout/per_category_accuracy/{sub_exp_id}": holdout_per_category_acc,
            f"holdout/per_group_accuracy/{sub_exp_id}": holdout_per_group_acc,
        }
        wandb.log(wandb_log_data)
