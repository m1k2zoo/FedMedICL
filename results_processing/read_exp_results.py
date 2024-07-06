import os, sys

parent_dir = os.path.dirname(os.getcwd())
sys.path.append(parent_dir)
os.chdir(parent_dir)

import numpy as np
import matplotlib.pyplot as plt

# import scienceplots
import seaborn as sns
from matplotlib.ticker import MaxNLocator, NullLocator
from reporting.util import compute_all_clients_model_accuracy


metric_name = "test_per_category_acc"

# Examples of experiment paths, modify this as needed
experiment_paths = [
    # "outputs/fitzpatrick17k_4T_10C_150R_5I_CB_group_probability",
    # "outputs/PAPILA_4T_10C_150R_5I_ERM_group_probability",
    # "outputs/HAM10000_4T_10C_150R_5I_mixup_group_probability",
]

if len(experiment_paths) == 0:
    print("You need to define at least one experiment path.")
else:
    for i, path in enumerate(experiment_paths):
        print("Path:", path)
        clients_accuracies = compute_all_clients_model_accuracy(path, metric_name)

        mean = np.mean(clients_accuracies, axis=0)
        std = np.std(clients_accuracies, axis=0)
        print(f"Mean {metric_name}: {mean[-1]:.2f} +- {std[-1]:.1f}\n")
