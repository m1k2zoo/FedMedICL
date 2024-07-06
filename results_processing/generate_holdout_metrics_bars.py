import os, sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
os.chdir(parent_dir)

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import scienceplots
import seaborn as sns
from matplotlib.ticker import MaxNLocator, NullLocator
from reporting.util import compute_all_clients_model_accuracy

# Setup argument parser for the script
parser = argparse.ArgumentParser(description="Generate holdout metrics plots for various datasets.")
parser.add_argument(
    "--algorithms",
    nargs="+",
    default=["ERM", "fedavg", "mixup", "SWAD", "ER", "resampling", "CRT", "CB"],
    help="Specify one or more algorithms to use. Defaults include various options.",
)
parser.add_argument(
    "--datasets",
    nargs="+",
    default=["COVID", "CheXpert", "fitzpatrick17k", "HAM10000", "PAPILA", "OL3I"],
    help="Specify which datasets to process. Defaults to all available datasets.",
)
parser.add_argument(
    "--exp_base_path",
    default="outputs",
    help="Specify the base path for experiment outputs. Default is 'outputs'.",
)
parser.add_argument(
    "--metric_name",
    default="holdout_per_category_acc",
    help="Specify the metric to be used. Default is 'holdout_per_category_acc'.",
)
parser.add_argument(
    "--num_tasks", type=int, default=4, help="Specify the number of tasks. Default is 4."
)
parser.add_argument(
    "--num_clients",
    type=int,
    default=10,
    help="Specify the number of clients for most datasets. Default is 10.",
)
parser.add_argument(
    "--num_chexpert_clients",
    type=int,
    default=50,
    help="Specify the number of clients for the CheXpert dataset. Default is 50.",
)
parser.add_argument(
    "--num_rounds", type=int, default=150, help="Specify the number of rounds. Default is 150."
)
parser.add_argument(
    "--num_iters", type=int, default=5, help="Specify the number of iterations. Default is 5."
)

args = parser.parse_args()


# Calculate averages, minimum and maximum deviations
def calc_avg_min_max(acc_list):
    deviation_threshold = 0.01
    avg = np.mean(acc_list)
    min_acc = np.min(acc_list)
    max_acc = np.max(acc_list)

    # Calculate min deviation if significant
    min_deviation = avg - min_acc
    if abs(min_deviation) <= deviation_threshold:
        min_deviation = 0

    # Calculate max deviation if significant
    max_deviation = max_acc - avg
    if abs(max_deviation) <= deviation_threshold:
        max_deviation = 0

    return avg, min_deviation, max_deviation


def process_and_plot_accuracies(ax, algorithm_data, algorithm_name, bar_width, config, x_offset):
    bar_positions = []
    color = config[algorithm_name]["color_bar"]
    label = config[algorithm_name]["legend"]

    position = x_offset

    if algorithm_data:
        avg, min_dev, max_dev = calc_avg_min_max(algorithm_data)
        ax.bar(
            position,
            avg,
            bar_width,
            # yerr=[[min_dev], [max_dev]] if max_dev > 0 else None,
            label=label,
            color=color,
            capsize=0,
            zorder=3,
            edgecolor="grey",
            linewidth=1,
        )
    bar_positions.append(position)
    return bar_positions


def rename_datasets_for_display(dataset_names):
    display_names = []
    for name in dataset_names:
        if name == "COVID":
            display_names.append("CheXCOVID")
        elif name == "fitzpatrick17k":
            display_names.append("Fitzpatrick17k")
        else:
            display_names.append(name)
    return display_names


# plt.style.use("science")
config_path = "results_processing/plot_style_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

algorithm_list = args.algorithms
dataset_list = args.datasets

bar_width = 0.15
fig_width = max(15, len(dataset_list) * 2.5)
fig, ax = plt.subplots(figsize=(fig_width, 3), constrained_layout=True)
xticks_font_size = 22
ylabel_font_size = 22
yticks_font_size = 20
legend_font_size = 16

# Calculate the index position of each dataset results
space_between_datasets = (bar_width * len(algorithm_list)) / 1.7
index = np.arange(len(dataset_list)) + space_between_datasets * np.arange(len(dataset_list))

x_tick_positions = []
for dataset_index, dataset_name in enumerate(dataset_list):
    clients = args.num_chexpert_clients if dataset_name == "CheXpert" else args.num_clients
    x_offset = index[dataset_index]
    bar_positions = []

    for algorithm in algorithm_list:
        path = f"{args.exp_base_path}/{dataset_name}_{args.num_tasks}T_{clients}C_{args.num_rounds}R_{args.num_iters}I_{algorithm}_group_probability"
        algorithm_data = compute_all_clients_model_accuracy(path, args.metric_name)

        if algorithm_data:
            avg, min_dev, max_dev = calc_avg_min_max(algorithm_data)
            ax.bar(
                x_offset,
                avg,
                bar_width,
                label=config[algorithm]["legend"],
                color=config[algorithm]["color_bar"],
                capsize=0,
                zorder=3,
                edgecolor="grey",
                linewidth=1,
            )
        bar_positions.append(x_offset)
        x_offset += bar_width

    # Center x-ticks between the bars of each dataset
    dataset_center = (bar_positions[0] + bar_positions[-1]) / 2
    x_tick_positions.append(dataset_center)


display_dataset_names = rename_datasets_for_display(dataset_list)

# Configure plot aesthetics
ax.set_ylabel("LTR Accuracy", fontsize=ylabel_font_size)
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(display_dataset_names)
ax.set_ylim(20, 80)
ax.yaxis.grid(True, linestyle="--", color="gray", alpha=0.5, zorder=0)


# Remove the upper and lower x-axis ticks
ax.tick_params(axis="x", which="both", top=False, bottom=False, labelsize=xticks_font_size)
# Disable x-axis ticks
ax.tick_params(axis="y", labelsize=yticks_font_size)


# Adjust the legend to display outside the plot
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))  # Remove duplicates

# Iterate over the algorithms specified in args, sort handles and labels according to the custom labels in config
sorted_handles = []
sorted_labels = []
for algorithm in algorithm_list:
    custom_label = config[algorithm]["legend"]
    if custom_label in unique_labels:
        sorted_handles.append(unique_labels[custom_label])
        sorted_labels.append(custom_label)

ax.legend(
    sorted_handles,
    sorted_labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.5),
    ncol=len(sorted_labels),
    columnspacing=2,
    fontsize=legend_font_size,
)

# Define the path to save the plot
save_path = "results_processing/main_plots"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
file_path = os.path.join(save_path, "holdout_metrics_bars.png")

# Save the plot
plt.savefig(file_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved at {file_path}")
