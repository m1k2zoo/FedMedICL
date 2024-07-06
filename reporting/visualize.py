import traceback
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from PIL import Image
import os
import copy
from reporting.util import (
    read_csv_results,
    aggregate_metric_over_clients,
    construct_matrix_from_dataframe,
    compute_cumulative_task_accuracy,
)


def plot_matrices_in_grid(model_matrices, titles, row_label, plot_dir, prefix="", cmap="RdBu"):
    """
    Plots matrices for multiple models in a grid layout and save them.

    Args:
    - model_matrices (list of list of pd.DataFrame): List containing matrices for each model.
    - titles (list of str): Titles for the columns (usually representing metrics).
    -  (str):row_label Label for the row (usually representing model name).
    - plot_dir (str): Directory where the plot should be saved.
    - prefix (str): A prefix for the filename.
    - cmap (str): Color map for the heatmaps.

    Returns:
    None
    """

    num_rows = len(model_matrices)
    num_cols = len(titles)

    plt.figure(figsize=(15, 5 * num_rows))
    for row_idx, matrices in enumerate(model_matrices):
        for col_idx, matrix in enumerate(matrices):
            if matrix is not None:
                plt.subplot(num_rows, num_cols, row_idx * num_cols + col_idx + 1)
                sns.heatmap(matrix, annot=True, cmap=cmap, cbar=True, square=True, fmt=".2f")
                plt.title(titles[col_idx])
                if col_idx == 0:
                    plt.figtext(
                        0.04,
                        1 - (row_idx * 1 / num_rows + 0.05 / num_rows),
                        row_label,
                        fontsize=16,
                        weight="bold",
                        color="Black",
                        ha="center",
                    )
    plt.tight_layout()

    filename = f"{prefix}_matrices_grid.png"
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300)  # You can adjust the dpi (dots per inch) as needed
    plt.close()


def plot_CL_metrics_from_logs(output_dir, split, algorithm, prefix="", wandb=None):
    """
    Plot task accuracies and losses from logs for all clients.

    Args:
        output_dir (str): The parent directory for the experiment outputs.
        split (str): The split to compute the metrics for (val, test or holdout).
        algorithm (str): The name of the algorithm.
        prefix (str): A prefix for the filename.
    """
    try:
        if split not in ["val", "test", "holdout"]:
            raise ValueError("Invalid split. Expected 'val', 'test' or 'holdout'.")

        # Create the output directory if it doesn't exist
        log_dir = os.path.join(output_dir, "logs")
        plot_dir = os.path.join(output_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)

        metrics_file_path = os.path.join(output_dir, f"final_CL_accuracies.txt")
        metrics = [f"{split}_acc", f"{split}_per_category_acc", f"{split}_per_group_acc"]
        if split in ["val", "test"]:
            # Read evaluation CSV results
            all_matrices = []
            eval_df = read_csv_results(log_dir, split)
            eval_matrices = []

            with open(metrics_file_path, "a") as metrics_file:
                metrics_file.write(f"{split} =========================:\n")
                for metric in metrics:
                    matrix = construct_matrix_from_dataframe(metric, eval_df)
                    metric_name = metric.replace(f"{split}_", "")
                    if matrix is not None:
                        eval_matrices.append(matrix)
                        final_accuracy = compute_cumulative_task_accuracy(matrix)[-1]
                        if metric_name == "per_category_acc":
                            metrics_file.write(f"{metric_name}: {final_accuracy:.2f}\n")

                        if wandb:
                            wandb.run.summary[f"{split}/final_CL_{metric_name}"] = final_accuracy

            if len(eval_matrices) > 0:
                all_matrices.append(eval_matrices)

            titles = [
                f"{split} Accuracy",
                f"{split} Per-Category Accuracy",
                f"{split} Per-Group Accuracy",
            ]
            prefix = f"{split}_{prefix}"
            plot_matrices_in_grid(all_matrices, titles, algorithm, plot_dir, prefix)

        else:
            fig, axes = plt.subplots(1, 3, figsize=(24, 5))  # 1 row, 3 columns
            for metric_column, ax in zip(metrics, axes):
                metric_values = []
                holdout_df = read_csv_results(log_dir, "holdout")
                if holdout_df is not None:
                    metric_values.append(holdout_df[metric_column].values)

                ax.boxplot(metric_values, vert=False)
                ax.set_yticks(list(range(1, 2)))
                ax.set_yticklabels([algorithm])
                ax.set_title("Boxplot for " + metric_column)
                ax.set_xlabel(metric_column)

            plt.tight_layout()

            filename = f"holdout_{prefix}_results.png"
            filepath = os.path.join(plot_dir, filename)
            plt.savefig(filepath, dpi=300)  # You can adjust the dpi (dots per inch) as needed
            plt.show()
            plt.close()

    except Exception as e:
        print(f"An error occurred in plot_CL_metrics_from_logs(): {e}")
        traceback.print_exc()


def plot_metrics_all_clients_from_logs(output_dir, algorithm, prefix=""):
    """
    Plot task accuracies and losses from logs for all clients.

    Args:
        output_dir (str): The parent directory for the experiment outputs.
        algorithm (str): The name of the algorithm.
        prefix (str): A prefix for the filename.
    """

    try:
        # Set color maps for plotting
        blue_cmap = plt.get_cmap("Blues")
        green_cmap = plt.get_cmap("Greens")
        red_cmap = plt.get_cmap("Reds")
        orange_cmap = plt.get_cmap("Oranges")
        JT_dark_color = "#352F44"
        JT_light_color = "#B9B4C7"

        # Create the output directory if it doesn't exist
        log_dir = os.path.join(output_dir, "logs")
        plot_dir = os.path.join(output_dir, "plot")
        os.makedirs(plot_dir, exist_ok=True)

        # Read CSV results for training and evaluation
        JT_train_df = read_csv_results(log_dir, "training", "JT")
        train_df = read_csv_results(log_dir, "training", algorithm)
        JT_eval_df = read_csv_results(log_dir, "evaluation", "JT")
        eval_df = read_csv_results(log_dir, "evaluation", algorithm)

        # Filter the dataframes to retain only rows where the training task ID matches the evaluation task ID
        if JT_train_df is not None:
            JT_eval_df = JT_eval_df[JT_eval_df["train_task_id"] == JT_eval_df["eval_task_id"]]
        eval_df = eval_df[eval_df["train_task_id"] == eval_df["eval_task_id"]]

        # Create a figure with three subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Since we're interested in the values from the last iteration, we retrieve the last row.
        # TODO: decide if this is what we want
        if JT_train_df is not None:  # Check if there's a joint training baseline
            last_JT_metrics = JT_eval_df.iloc[-1]
            JT_test_acc = last_JT_metrics["test_acc"]
            JT_test_category_acc = last_JT_metrics["test_per_category_acc"]
            JT_test_group_acc = last_JT_metrics["test_per_group_acc"]

        # test accuracy
        x = eval_df.iteration
        test_acc = eval_df.test_acc
        test_category_acc = eval_df.test_per_category_acc
        test_group_acc = eval_df.test_per_group_acc

        train_tasks_boundaries = train_df["iteration"][
            train_df["task_id"].shift(-1) - train_df["task_id"] == 1
        ].tolist()
        eval_tasks_boundaries = eval_df["iteration"][
            eval_df["train_task_id"].shift(-1) - eval_df["train_task_id"] == 1
        ].tolist()

        # Add horizontal lines for JT Test Acc and JT Test Category Acc
        if JT_train_df is not None:
            axs[0, 0].axhline(
                y=JT_test_acc, color=JT_light_color, linestyle="dashed", label="JT acc"
            )
            axs[0, 0].axhline(
                y=JT_test_category_acc,
                color=JT_dark_color,
                linestyle="-",
                label="JT per-category acc",
            )

        # Add task boundaries
        for boundary in eval_tasks_boundaries:
            axs[0, 0].axvline(x=boundary, color="gray", linestyle="dotted")

        axs[0, 0].plot(x, test_acc, linestyle="dashed", color=blue_cmap(0.4), label="acc")
        axs[0, 0].plot(
            x, test_category_acc, linestyle="-", color=blue_cmap(0.7), label="per-category acc"
        )

        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Average Accuracy")
        axs[0, 0].set_title("Per-Category Test Accuracy")
        axs[0, 0].legend()

        # Add horizontal lines for JT Test Acc and JT Test Group Acc
        if JT_train_df is not None:
            axs[0, 1].axhline(
                y=JT_test_acc, color=JT_light_color, linestyle="dashed", label="JT acc"
            )
            axs[0, 1].axhline(
                y=JT_test_group_acc, color=JT_dark_color, linestyle="-", label="JT per-group acc"
            )

        # Add task boundaries
        for boundary in eval_tasks_boundaries:
            axs[0, 1].axvline(x=boundary, color="gray", linestyle="dotted")

        axs[0, 1].plot(x, test_acc, linestyle="dashed", color=blue_cmap(0.4), label="acc")
        axs[0, 1].plot(
            x, test_group_acc, linestyle="-", color=blue_cmap(0.7), label="per-group acc"
        )

        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Average Accuracy")
        axs[0, 1].set_title("Per-Group Test Accuracy")
        axs[0, 1].legend()

        # Plot train and validation accuracy
        x1, train_mean_acc, y1_lower, y1_upper = aggregate_metric_over_clients(
            train_df, "train_acc"
        )

        # Add task boundaries
        for boundary in train_tasks_boundaries:
            axs[1, 0].axvline(x=boundary, color="gray", linestyle="dotted")

        axs[1, 0].plot(x1, train_mean_acc, color=blue_cmap(0.7), label="train acc")
        axs[1, 0].fill_between(x1, y1_lower, y1_upper, color=blue_cmap(0.7), alpha=0.2)

        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Average Accuracy")
        axs[1, 0].set_title("Train Accuracy")
        axs[1, 0].legend()

        # Plot train and validation losses
        x1, train_mean_loss, y1_lower, y1_upper = aggregate_metric_over_clients(
            train_df, "train_loss"
        )

        # Add task boundaries
        for boundary in train_tasks_boundaries:
            axs[1, 1].axvline(x=boundary, color="gray", linestyle="dotted")

        axs[1, 1].plot(x1, train_mean_loss, color=blue_cmap(0.7), label="train loss")
        axs[1, 1].fill_between(x1, y1_lower, y1_upper, color=blue_cmap(0.7), alpha=0.2)

        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Average Loss")
        axs[1, 1].set_title("Train Loss")
        axs[1, 1].legend()

        # # Adjust spacing between the subplots
        fig.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(plot_dir, f"{prefix}_results.png"))
    except Exception as e:
        print(f"An error occurred in plot_metrics_all_clients_from_logs(): {e}")
        traceback.print_exc()


def plot_metrics_all_clients(baseline_trackers, client_trackers, output_dir, prefix=""):
    """
    Plot the task accuracies and losses for multiple metrics trackers.

    Args:
        baseline_trackers (list): A list of baseline MetricsTracker.
        client_trackers (list): A list of main experiment MetricsTracker.
        output_dir (str): The parent directory for the experiment outputs.
        prefix (str): A prefix for the filename.
    """
    # Create the output directory if it doesn't exist
    plot_dir = os.path.join(output_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    for i in range(len(client_trackers)):
        plot_metrics_client(baseline_trackers[i], client_trackers[i], i, plot_dir, prefix)


def plot_metrics_client(baseline_tracker, metrics_tracker, client_index, output_dir, prefix=""):
    """
    Plots the baseline train and test accuracies, CL train and test accuracies,
    and CL train and validation losses over all tasks.

    Args:
        baseline_tracker (MetricsTracker): An object that tracks metrics for a specific client baseline.
        metrics_tracker (MetricsTracker): An object that tracks metrics for a specific client.
        client_index (int): The index of the client.
        output_dir (str): The directory to save the plot.
        prefix (str): A prefix for the filename.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    blue_cmap = plt.get_cmap("Blues")
    green_cmap = plt.get_cmap("Greens")
    red_cmap = plt.get_cmap("Reds")
    orange_cmap = plt.get_cmap("Oranges")

    # Get the baseline train and test accuracies
    baseline_test_accuracy, baseline_test_macro_accuracy, _ = baseline_tracker.average_acc(
        0, "test"
    )

    # Initialize lists to store tasks, train and test accuracies, and train and validation losses
    tasks = []
    task_train_accuracies = []
    task_train_macro_accuracies = []
    task_test_accuracies = []
    task_test_macro_accuracies = []
    task_train_losses = []
    task_val_losses = []

    # Loop over all task IDs
    for task_id in range(metrics_tracker.num_tasks):
        # Get the task train and test accuracies
        task_train_acc, task_train_macro_acc, _ = metrics_tracker.average_acc(task_id, "train")
        task_test_acc, task_test_macro_acc, _ = metrics_tracker.average_acc(task_id, "test")

        # Get the task train and validation losses
        task_train_loss = metrics_tracker.average_loss(task_id, split="train")
        task_val_loss = metrics_tracker.average_loss(task_id, split="val")

        # Append the task ID and accuracies/losses to the lists
        tasks.append(task_id + 1)
        task_train_accuracies.append(task_train_acc)
        task_train_macro_accuracies.append(task_train_macro_acc)
        task_test_accuracies.append(task_test_acc)
        task_test_macro_accuracies.append(task_test_macro_acc)
        task_train_losses.append(task_train_loss)
        task_val_losses.append(task_val_loss)

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Plot the baseline train and test accuracies as horizontal lines
    axs[0].axhline(
        baseline_test_accuracy, color=red_cmap(0.4), linestyle="--", label="Baseline Test Accuracy"
    )
    axs[0].axhline(
        baseline_test_macro_accuracy,
        color=red_cmap(0.7),
        linestyle="--",
        label="Baseline Test Macro Accuracy",
    )

    # Plot the task train and test accuracies
    # axs[0].plot(tasks, task_train_accuracies, marker='o', color=blue_cmap(0.4), label='CL Train Accuracy')
    # axs[0].plot(tasks, task_train_macro_accuracies, marker='o', color=blue_cmap(0.7), label='CL Train Macro Accuracy')
    axs[0].plot(
        tasks, task_test_accuracies, marker="o", color=green_cmap(0.4), label="CL Test Accuracy"
    )
    axs[0].plot(
        tasks,
        task_test_macro_accuracies,
        marker="o",
        color=green_cmap(0.7),
        label="CL Test Macro Accuracy",
    )

    # Customize the first plot
    axs[0].set_xlabel("Task")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Baseline and CL Train/Test Accuracies")
    axs[0].legend()

    # Plot the task train and validation losses
    axs[1].plot(tasks, task_train_losses, marker="o", color=blue_cmap(0.7), label="CL Train Loss")
    axs[1].plot(
        tasks, task_val_losses, marker="o", color=orange_cmap(0.7), label="CL Validation Loss"
    )

    # Customize the second plot
    axs[1].set_xlabel("Task")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("CL Train/Validation Losses")
    axs[1].legend()

    # Adjust spacing between the subplots
    fig.tight_layout()

    # Save the plot with client_index in the figure name
    plt.savefig(os.path.join(output_dir, f"{prefix}_accuracies_losses_client_{client_index}.png"))


def plot_multi_horizontal_bar(
    all_sizes, legends, title, subtitles, filename, parent_dir, visualize_by
):
    """
    Plots a horizontal bar chart of subpopulations.

    Parameters:
    all_sizes (list): An array representing the sizes of each subpopulation.
    legends (list): A list of strings representing the legends for the subpopulations.
    title (str): A title for the plot
    subtitles (list): A list of subtitles for each subpopulation
    filename (str): A filename for the saved plot image
    parent_dir (str): The parent directory for the saved plot image
    visualize_by (str): The visualization criterion. Can be 'group' or 'target'.

    Returns:
    None.
    """

    # Create a horizontal bar plot
    fig_height = 0.3 * len(all_sizes) * 2
    fig_height = max(2, fig_height)
    height_factor = 1 / fig_height
    fig, ax = plt.subplots(figsize=(8, fig_height))

    # Define a modern color palette
    colors = sns.color_palette("RdBu", len(legends))

    if len(colors) == 7:
        # Grey color
        colors[3] = (0.5, 0.5, 0.5)

    height = 0.2
    for i, sizes in enumerate(all_sizes):
        # Calculate the total population size
        total_population = sum(sizes)

        # Calculate the percentages of each subpopulation
        percentages = [size / total_population for size in sizes]

        # Plot the horizontal bar
        ypos = 0.5 - i / 4
        for j, (legend, percentage) in enumerate(zip(legends, percentages)):
            if i == 0:
                ax.barh(
                    ypos,
                    percentage,
                    height=height,
                    left=np.sum(percentages[:j]),
                    color=colors[j],
                    label=legend,
                )
            else:
                ax.barh(
                    ypos, percentage, height=height, left=np.sum(percentages[:j]), color=colors[j]
                )

            # Skip printing percentages smaller than 2%
            if percentage > 0.02:
                ax.text(
                    np.sum(percentages[:j]) + percentage / 2,
                    ypos,
                    f"{percentage:.0%}",
                    ha="center",
                    va="center",
                    color="black",
                )
        dataset_title = ax.text(
            -0.03, ypos, f"{subtitles[i]}", ha="center", va="center", color="black"
        )

    # Hide the x-axis and y-axis values and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set the x-axis limits
    ax.set_xlim([0, 1])

    # Set the title
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add a legend
    ypos_legend = -0.8 * height_factor
    legend_title = visualize_by.capitalize() + "s"
    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, ypos_legend),
        ncol=10,
        fancybox=True,
        shadow=True,
        title=legend_title,
    )

    # Add raw values for each sub-dataset
    values_text = ""
    dataset_rows = len(all_sizes)
    for i in range(dataset_rows):
        values_text += f"{subtitles[i]} values: {all_sizes[i]}"
        if i != dataset_rows - 1:
            values_text += "\n"

    ypos_text = ypos_legend - 0.22 * height_factor * (dataset_rows)
    raw_values_text = ax.text(0.5, ypos_text, values_text, transform=ax.transAxes, ha="center")

    # Save the plot
    Path(parent_dir).mkdir(parents=True, exist_ok=True)
    output_filename = os.path.join(parent_dir, filename)
    fig.savefig(
        output_filename,
        bbox_extra_artists=(dataset_title, legend, raw_values_text),
        bbox_inches="tight",
    )
    plt.close(fig)


def get_target_sizes(dataframe, target_index):
    """
    Get the sizes of target in a given DataFrame and return the sizes in the order of target_index.

    Parameters:
    - dataframe (pandas.Series): A pandas Series containing target data.
    - target_index (list): A list of target values to be included in the result.

    Returns:
    - target_sizes (numpy.ndarray): An array of target sizes corresponding to the order of target_index.
    """
    target_counts = dataframe.Target.value_counts().sort_index()
    target_counts = target_counts.reindex(target_index, fill_value=0)
    target_sizes = target_counts.values
    return target_sizes


def get_dataset_sizes_by_attribute(dataset, attribute):
    """
    Get the sizes of attribute groups in a given dataset.

    Parameters:
    dataset (BaseDataset): An instance of BaseDataset
    attribute (str): The visualization criterion. Can be 'group' or 'target'.

    Returns:
    tuple: A tuple containing the sizes of attribute and the legend labels.
    """

    if attribute not in ["group", "target"]:
        raise ValueError("Invalid value for attribute. Use 'group' or 'target'.")

    if attribute == "group":
        global_attribute_groups = dataset.all_attribute_groups
        local_attribute_groups = dataset.get_attribute_groups()

        legends = [str(attribute_group) for attribute_group in global_attribute_groups]
        attribute_sizes = [0] * len(global_attribute_groups)

        for attribute in local_attribute_groups:
            # Get the global index of the attribute
            attribute_global_index = global_attribute_groups.index(attribute.attribute_group)
            # Insert the attribute size in the global index
            attribute_sizes[attribute_global_index] = len(attribute)

    elif attribute == "target":
        legends = dataset.targets_set
        attribute_sizes = get_target_sizes(dataset.dataframe, legends)

    return attribute_sizes, legends


def visualize_full_dataset(dataset, filename, parent_dir, visualize_by):
    """
    Visualize the attribute groups distribution for the training, validation and testing sets of a dataset,
    and saves the visualization as a PNG file.

    Parameters:
    - dataset (DatasetSplit object): The dataset to visualize.
    - filename (str): The filename to save the visualization as.
    - parent_dir (str): The parent directory to save the visualization file in.
    - visualize_by (str): The visualization criterion. Can be 'group' or 'target'.

    Returns:
    None.
    """

    train_attribute_sizes, legends = get_dataset_sizes_by_attribute(dataset.train_set, visualize_by)
    val_attribute_sizes, _ = get_dataset_sizes_by_attribute(dataset.val_set, visualize_by)
    test_attribute_sizes, _ = get_dataset_sizes_by_attribute(dataset.test_set, visualize_by)
    holdout_attribute_sizes, _ = get_dataset_sizes_by_attribute(dataset.holdout_set, visualize_by)
    attribute_sizes = [
        train_attribute_sizes,
        val_attribute_sizes,
        test_attribute_sizes,
        holdout_attribute_sizes,
    ]

    title = f"Dataset_{dataset.name}"
    subtitles = ["Train", "Val", "Test", "Holdout"]
    plot_multi_horizontal_bar(
        attribute_sizes, legends, title, subtitles, filename, parent_dir, visualize_by
    )


def visualize_all_datasets(datasets_list, parent_dir):
    """
    Generates a visualization of the attribute groups distribution for each dataset and merge all
    visualizations into a single image.

    Parameters:
    - datasets_list (list of DatasetSplit): A list of DatasetSplit objects to visualize.
    - parent_dir (str): The parent directory to save the visualization file in.

    Returns:
    None.
    """

    for visualize_by in ["group", "target"]:
        # Generate individual visualizations for each dataset and save as separate PNG files
        filenames = []
        for dataset in datasets_list:
            filename = dataset.name + ".png"
            filenames.append(os.path.join(parent_dir, filename))
            visualize_full_dataset(dataset, filename, parent_dir, visualize_by)

        # Combine all individual plots into a single plot
        combine_plots(filenames, os.path.join(parent_dir, f"all_datasets_by_{visualize_by}.png"))


def visualize_dataset_clients(full_dataset, clients_splits, filename, parent_dir, visualize_by):
    """
    Visualize the attribute groups distribution for the training sets of a dataset and
    its client splits, and saves the visualization as a PNG file.

    Parameters:
    - full_dataset (DatasetSplit object): The full dataset to visualize.
    - clients_splits (List of DatasetSplit objects): The list of client splits to visualize.
    - filename (str): The filename to save the visualization as.
    - parent_dir (str): The parent directory to save the visualization file in.
    - visualize_by (str): The visualization criterion. Can be 'group' or 'target'.

    Returns:
    None.
    """
    sizes = []
    subtitles = []

    full_dataset_attribute_sizes, legends = get_dataset_sizes_by_attribute(
        full_dataset.train_set, visualize_by
    )
    sizes.append(full_dataset_attribute_sizes)
    subtitles.append("Full")

    for split in clients_splits:
        client_sizes, _ = get_dataset_sizes_by_attribute(split.train_set, visualize_by)
        sizes.append(client_sizes)
        subtitles.append(f"C{split.client_id}")

    title = f"Clients_of_{full_dataset.name}"
    plot_multi_horizontal_bar(sizes, legends, title, subtitles, filename, parent_dir, visualize_by)


def visualize_all_clients(full_datasets_list, client_dataset_list, parent_dir):
    """
    Generates a visualization of the attribute groups distribution for each full dataset and its client splits,
    and merges all visualizations into a single image.

    Parameters:
    - full_datasets_list (list of DatasetSplit objects): A list of full datasets to visualize.
    - client_dataset_list (list of DatasetSplit objects): A list of client datasets to visualize.
    - parent_dir (str): The parent directory to save the visualization file in.

    Returns:
    None.
    """

    for visualize_by in ["group", "target"]:
        filenames = []
        # Generate visualizations for the clients of each dataset and save as separate PNG files
        for dataset in full_datasets_list:
            filename = f"clients_of_{dataset.name}.png"
            filenames.append(os.path.join(parent_dir, filename))

            clients_splits = [
                client_dataset
                for client_dataset in client_dataset_list
                if client_dataset.name == dataset.name
            ]
            visualize_dataset_clients(dataset, clients_splits, filename, parent_dir, visualize_by)

        # Combine all individual plots into a single plot
        combine_plots(filenames, os.path.join(parent_dir, f"all_clients_by_{visualize_by}.png"))


def visualize_client_tasks(full_dataset, client_split, split, filename, parent_dir, visualize_by):
    """
    Visualizes the attribute group distribution for each task in a client split of a dataset,
    and saves the visualization as a PNG file.

    Parameters:
    - full_dataset (DatasetSplit object): The full dataset to which the client split belongs.
    - client_split (DatasetSplit objects): The client split to visualize.
    - split (str): The split set specified by the split parameter. Must be: 'train', 'val', or 'test'.
    - filename (str): The filename to save the visualization as.
    - parent_dir (str): The parent directory to save the visualization file in.
    - visualize_by (str): The visualization criterion. Can be 'group' or 'target'.

    Returns:
    None.
    """

    def get_split_set(dataset, split):
        """
        Returns the specified split set from a given dataset.
        The split parameter must be one of the following values: 'train', 'val', or 'test'.

        Parameters:
        - dataset (DatasetSplit object): The dataset from which to retrieve the split set.
        - split (str): The split set specified by the split parameter.

        Returns:
        The split set specified by the split parameter.

        """
        if split == "train":
            return dataset.train_set
        elif split == "val":
            return dataset.val_set
        elif split == "test":
            return dataset.test_set
        else:
            raise ValueError("Invalid split. Expected 'train', 'val', or 'test'.")

    client_split = copy.deepcopy(client_split)
    attribute_sizes = []
    subtitles = []

    # Get the attribute distribution for the full dataset
    full_dataset_attribute_sizes, legends = get_dataset_sizes_by_attribute(
        get_split_set(full_dataset, split), visualize_by
    )
    attribute_sizes.append(full_dataset_attribute_sizes)
    subtitles.append("Full")

    # Get the attribute distribution for the client split
    client_dataset = get_split_set(client_split, split)
    client_attribute_sizes, _ = get_dataset_sizes_by_attribute(client_dataset, visualize_by)
    attribute_sizes.append(client_attribute_sizes)
    subtitles.append(f"C{client_split.client_id}")

    # Get the attribute distribution for each task in the client split
    for task_id in range(client_split.num_tasks):
        is_train = True if split == "train" else False
        client_dataset.load_task(task_id, client_split.num_tasks, is_train=is_train, is_print=False)
        task_attribute_sizes, _ = get_dataset_sizes_by_attribute(
            get_split_set(client_split, split), visualize_by
        )
        attribute_sizes.append(task_attribute_sizes)
        subtitles.append(f"T{task_id}")

    title = f"Client_{client_split.client_id}_{split}_Tasks_({full_dataset.name})"

    plot_multi_horizontal_bar(
        attribute_sizes, legends, title, subtitles, filename, parent_dir, visualize_by
    )


def visualize_all_clients_tasks(full_datasets_list, client_dataset_list, parent_dir):
    """
    Generates a visualization of the attribute group distribution for each task in each
    client split, and merges all visualizations into a single image.

    Parameters:
    - full_datasets_list (list of DatasetSplit objects): A list of full datasets to visualize.
    - client_dataset_list (list of DatasetSplit objects): A list of client splits to visualize.
    - parent_dir (str): The parent directory to save the visualization file in.

    Returns:
    None.
    """
    for visualize_by in ["group", "target"]:
        full_datasets_names = [dataset.name for dataset in full_datasets_list]

        # Generate visualizations for the tasks of each client split
        for client_split in client_dataset_list:
            filenames = []
            client_id = client_split.client_id
            if client_id == -1:
                client_id = "JT"

            # Get the full dataset that of this client
            full_dataset = full_datasets_list[full_datasets_names.index(client_split.name)]

            for split_name in ["train", "val", "test"]:
                filename = f"client_{client_id}_{split_name}_tasks.png"
                filenames.append(os.path.join(parent_dir, filename))
                visualize_client_tasks(
                    full_dataset, client_split, split_name, filename, parent_dir, visualize_by
                )

            combine_plots(
                filenames,
                os.path.join(parent_dir, f"client_{client_id}_tasks_by_{visualize_by}.png"),
                vertical=False,
            )


def combine_plots(image_paths, filename, remove_old=True, vertical=True):
    """
    Combines multiple PNG plots into a single image file, arranged in a grid layout.

    Parameters:
    - image_paths (list): A list of filepaths to the PNG image files to combine.
    - filename (str): The filename to save the combined image as.
    - remove_old (boolean): A flag to decide whether to remove individual images
    - vertical (boolean): A flag to decide whether to combine plots vertically or horizontally

    Returns:
    None.
    """

    if len(image_paths) == 1:
        # If only one plot, rename the file to the given filename and return without combining plots
        os.rename(image_paths[0], filename)
        return

    # Set the desired grid layout based on the number of plots
    num_plots = len(image_paths)

    # Get the width and height of each plot
    img_heights = []
    img_widths = []
    for path in image_paths:
        img = Image.open(path)
        img_widths.append(img.size[0])
        img_heights.append(img.size[1])
        img.close()

    # Calculate the size of the combined image
    if vertical:
        new_img_width = max(img_widths)
        new_img_height = sum(img_heights)
    else:
        new_img_width = sum(img_widths)
        new_img_height = max(img_heights)

    new_img = Image.new("RGB", (new_img_width, new_img_height), color=(255, 255, 255))

    # Loop through each plot and paste it onto the new image
    for i in range(num_plots):
        img = Image.open(image_paths[i])

        # Calculate the paste position based on the individual heights and widths
        if vertical:
            paste_x = 0
            paste_y = sum(img_heights[:i])
        else:
            paste_x = img_widths[i] * i
            paste_y = 0

        # Paste the plot onto the new image
        new_img.paste(img, (paste_x, paste_y))

    # Save the new image with the specified filename
    new_img.save(filename)

    # Remove individual PNG files for individual dataset visualizations
    if remove_old:
        for plot_image in image_paths:
            os.remove(plot_image)
