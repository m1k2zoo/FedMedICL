import os, sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
os.chdir(parent_dir)

import argparse
import traceback
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import scienceplots
import json
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.ticker import MultipleLocator
from reporting.util import compute_novel_disease_results


parser = argparse.ArgumentParser(description="Process the parameters for the experiment.")

# Algorithm type
parser.add_argument(
    "--algorithms",
    nargs="+",  # Allows multiple inputs
    default=["ERM", "fedavg", "CB", "resampling", "ER"],
    help="Specify one or more algorithms to use. Defaults to all algorithms ['ERM', 'fedavg', 'CB', 'resampling', 'ER'].",
)

# Experiment base path
parser.add_argument(
    "--exp_base_path",
    default="outputs",
    help="Specify the base path for experiment outputs. Default is 'outputs'.",
)

# Metric name
parser.add_argument(
    "--metric_name",
    default="holdout_per_category_acc",
    help="Specify the name of the metric to be used. Default is 'holdout_per_category_acc'.",
)

# Number of tasks
parser.add_argument(
    "--num_tasks", type=int, default=4, help="Specify the number of tasks. Default is 4."
)

# Number of clients
parser.add_argument(
    "--num_clients", type=int, default=5, help="Specify the number of clients. Default is 5."
)

# Number of rounds
parser.add_argument(
    "--num_rounds", type=int, default=150, help="Specify the number of rounds. Default is 150."
)

# Number of iterations
parser.add_argument(
    "--num_iters", type=int, default=5, help="Specify the number of iterations. Default is 5."
)


args = parser.parse_args()
metric_name = args.metric_name + "_dict"

# plt.style.use('science')
config_path = "results_processing/plot_style_config.json"
with open(config_path, "r") as config_file:
    config = json.load(config_file)

dpi = 300
y_min = 15.0
y_max = 65.0

title_font_size = 20
xlabel_font_size = 22
ylabel_font_size = 22
ticks_font_size = 18
legend_font_size = 18

line_width = 3
marker_size = 7
alpha_value = 0.8

fig, axs = plt.subplots(1, 2, figsize=(10, 3.5), dpi=dpi)

legends = []
for algorithm_name in args.algorithms:
    exp_path = f"{args.exp_base_path}/COVID_{args.num_tasks}T_{args.num_clients}C_{args.num_rounds}R_{args.num_iters}I_{algorithm_name}_Naive_novel/logs"
    try:
        (
            common_acc_mean,
            common_acc_std,
            novel_acc_mean,
            novel_acc_std,
        ) = compute_novel_disease_results(exp_path, algorithm_name, metric_name)
        tasks = range(1, len(common_acc_mean) + 1)
        color = config[algorithm_name]["color"]
        legend = config[algorithm_name]["legend"]
        legends.append(legend)
        line_style = "--" if algorithm_name == "ERM" else "-"
        axs[0].plot(
            tasks,
            common_acc_mean,
            label=legend,
            marker="o",
            color=color,
            linewidth=line_width,
            markersize=marker_size,
            alpha=alpha_value,
            linestyle=line_style,
        )
        axs[1].plot(
            tasks,
            novel_acc_mean,
            label=legend,
            marker="o",
            color=color,
            linewidth=line_width,
            markersize=marker_size,
            alpha=alpha_value,
            linestyle=line_style,
        )
    except FileNotFoundError as e:
        print(f"Error: The file or directory was not found: {e}")
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

if len(legends) == 0:
    raise Exception(
        "\nCouldn't load any of the experiment results. Make sure the arguments are correct."
    )

# Setting up the common accuracies plot
axs[0].set_title("Non-COVID Performance", fontsize=title_font_size)
axs[0].set_xlabel("Task", fontsize=xlabel_font_size)
axs[0].set_ylabel("LTR Accuracy", fontsize=ylabel_font_size)
axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[0].xaxis.set_minor_locator(NullLocator())
axs[0].yaxis.set_major_locator(MultipleLocator(10))
axs[0].tick_params(axis="both", labelsize=ticks_font_size)
axs[0].grid(True)

# Setting up the novel accuracies plot
axs[1].set_title("COVID Performance", fontsize=title_font_size)
axs[1].set_xlabel("Task", fontsize=xlabel_font_size)
axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
axs[1].xaxis.set_minor_locator(NullLocator())
axs[1].yaxis.set_major_locator(MultipleLocator(25))
axs[1].tick_params(axis="both", labelsize=ticks_font_size)
axs[1].grid(True)

fig.legend(
    legends,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.07),
    ncol=5,
    columnspacing=1.5,
    fontsize=legend_font_size,
)
fig.tight_layout(pad=2.0)

# Define the path to save the plot
save_path = "results_processing/novel_plots"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist
file_path = os.path.join(save_path, "novel_performance.png")

# Save the plot
plt.savefig(file_path, dpi=dpi, bbox_inches="tight")
print(f"\nPlot saved at {file_path}")
