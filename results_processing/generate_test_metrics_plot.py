import os, sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
os.chdir(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

# import scienceplots
import seaborn as sns
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.ticker import MultipleLocator
from reporting.util import compute_all_clients_model_accuracy
import json
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="Generate test metrics plots for various datasets.")
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
    default="test_per_category_acc",
    help="Specify the metric to be used. Default is 'test_per_category_acc'.",
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


# Mapping dataset names to their respective titles
dataset_title_map = {
    "COVID": "CheXCOVID(IF=10.8)",
    "CheXpert": "CheXpert(IF=9.6)",
    "fitzpatrick17k": "Fitzpatrick17k(IF=5.4)",
    "HAM10000": "HAM10000(IF=58.3)",
    "PAPILA": "PAPILA(IF=5.2)",
    "OL3I": "OL3I(IF=22.1)",
}


dataset_names = args.datasets
dataset_titles = [dataset_title_map[name] for name in dataset_names if name in dataset_title_map]
num_tasks = args.num_tasks
rounds = args.num_rounds
iterations = args.num_iters

# plt.style.use("science")
metric_name = "test_per_category_acc"
config_path = "results_processing/plot_style_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

title_font_size = 20
xlabel_font_size = 22
ylabel_font_size = 22
ticks_font_size = 18
legend_font_size = 18

y_min = 15.0
y_max = 65.0
line_width = 2.5
marker_size = 7
alpha_value = 0.9  # Transparency for lines


fig, axs = plt.subplots(1, len(dataset_names), figsize=(int(4 * len(dataset_names)), 4))
x = range(1, num_tasks + 1)
plot_counts = {alg: 0 for alg in args.algorithms}

for i, dataset_name in enumerate(dataset_names):
    print(dataset_name)
    clients = args.num_chexpert_clients if dataset_name == "CheXpert" else args.num_clients

    ax = axs if len(dataset_names) == 1 else axs[i]

    # Get accuracy values for the current dataset
    for algorithm in args.algorithms:
        # Construct the exp path based on the current dataset
        path = f"{args.exp_base_path}/{dataset_name}_{num_tasks}T_{clients}C_{rounds}R_{iterations}I_{algorithm}_group_probability"

        if os.path.exists(path):
            clients_accs = compute_all_clients_model_accuracy(path, metric_name)
            if clients_accs:
                algorithms_mean = np.mean(clients_accs, axis=0)
                ax.plot(
                    x,
                    algorithms_mean,
                    label=config[algorithm]["legend"],
                    marker="o",
                    color=config[algorithm]["color"],
                    linewidth=line_width,
                    markersize=marker_size,
                    alpha=alpha_value,
                    linestyle="--" if algorithm == "ERM" else "-",
                )
                plot_counts[algorithm] += 1
        else:
            print(f"Path does not exist: {path}. Skipping {algorithm} for {dataset_name}.")

    ax.set_title(dataset_titles[i], fontsize=title_font_size)
    ax.set_xlabel("Task", fontsize=xlabel_font_size)
    if i == 0:
        ax.set_ylabel("LTR over seen tasks", fontsize=ylabel_font_size)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if dataset_name in ["CheXpert", "fitzpatrick17k", "PAPILA"]:
        ax.yaxis.set_major_locator(MultipleLocator(5))
    if dataset_name == "OL3I":
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", labelsize=ticks_font_size)
    ax.grid(True)


if len(dataset_names) == 1:
    handles, labels = axs.get_legend_handles_labels()
else:
    # Determine which axes to use for the legend by finding the dataset with the most plotted algorithms
    most_plotted_dataset_index = np.argmax([plot_counts[alg] for alg in args.algorithms])
    handles, labels = axs[most_plotted_dataset_index].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.1),
    ncol=len(labels),
    columnspacing=2,
    fontsize=legend_font_size,
)
fig.tight_layout(pad=2.0)
# fig.subplots_adjust(bottom=0.25)


# Define the path to save the plot
save_path = "results_processing/main_plots"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
file_path = os.path.join(save_path, "test_metrics.png")

# Save the plot
plt.savefig(file_path, dpi=300, bbox_inches="tight")
print(f"\nPlot saved at {file_path}")
