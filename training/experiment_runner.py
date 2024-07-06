from training.util import set_seed, initialize_models_and_optimizers, initialize_client_trackers
from training.train_management import train_centralized, train_all_clients
from reporting.visualize import plot_metrics_all_clients_from_logs, plot_CL_metrics_from_logs
from reporting.logging import log_summary_metrics_to_wandb
from helpers.client_management import save_objects
from reporting.util import compute_novel_disease_results

import os
import copy
import time
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau


def modify_args_for_baseline(args, is_joint_training=False):
    """
    Modify the arguments based on whether the training is joint or local.

    """
    new_args = copy.deepcopy(args)
    # new_args.num_tasks = 1 # Disable continual task splitting
    if is_joint_training:
        new_args.num_rounds = (
            args.num_evaluations
        )  # Use rounds to allow for intermidate evaluations
        new_args.num_iters = int(args.jt_iters / args.num_evaluations)
        new_args.num_clients = 1
    return new_args


def run_joint_baseline(args, dataset, best_lr_dict, wandb):
    """
    Run the joint training baseline model. This function sets up the conditions for
    joint training and uses a common function to execute the training.

    Args:
    args (argparse.Namespace): Parsed command-line arguments.
    dataset (DatasetSplit): The dataset to perform joint training on.
    best_lr_dict (dict): Dictionary with best learning rates.
    wandb (wandb): Instance of wandb logger.

    """
    start_time = time.time()

    set_seed(args.seed)
    is_joint_training = True

    # Modify args based on the type of training
    modified_args = modify_args_for_baseline(args, is_joint_training)

    baseline_trackers = initialize_client_trackers(modified_args, client_dataset_list, algorithm)
    models, optimizers = initialize_models_and_optimizers(
        modified_args, client_dataset_list, best_lr_dict
    )

    if is_joint_training:
        patience = 200
        if client_dataset_list[0].name == "OL3I":
            patience = 20  # Since OL3I is exteremly small, we need a smaller patience
        # jt_scheduler = ReduceLROnPlateau(optimizers[0], factor=0.5, patience=patience, threshold=0.0001)
        jt_scheduler = None
    else:
        jt_scheduler = None

    use_fl = False  # Disable federated learning
    baseline_trackers = train_centralized(
        models,
        optimizers,
        client_dataset_list,
        baseline_trackers,
        modified_args,
        use_fl,
        wandb,
        jt_scheduler,
    )

    # Save the trackers
    filename = f"{algorithm}_baseline_trackers.pkl"
    save_objects(baseline_trackers, os.path.join(args.output_dir, filename))

    end_time = time.time()
    duration_seconds = end_time - start_time
    duration_hours = duration_seconds / 3600  # Convert seconds to hours
    print(f"Baseline experiment completed in {duration_hours:.2f} hours")


def run_main_experiment(args, client_dataset_list, best_lr_dict, wandb):
    """
    Train the models on each client's dataset with continual and federated learning.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        client_dataset_list (list): List of client datasets.
        best_lr_dict (dict): lr dictionary to initialize the optimizers.
        wandb (wandb): Instance of wandb logger.

    """
    start_time = time.time()

    use_cl = args.num_tasks > 1
    use_fl = args.use_fl

    set_seed(args.seed)
    client_trackers = initialize_client_trackers(args, client_dataset_list, args.algorithm)
    models, optimizers = initialize_models_and_optimizers(args, client_dataset_list, best_lr_dict)

    exp_name = "exp"

    client_trackers = train_all_clients(
        models, optimizers, client_dataset_list, client_trackers, args, use_fl, wandb
    )

    # plot_metrics_all_clients_from_logs(args.output_dir, args.algorithm, exp_name)
    plot_CL_metrics_from_logs(args.output_dir, "test", args.algorithm, exp_name, wandb)
    plot_CL_metrics_from_logs(args.output_dir, "val", args.algorithm, exp_name, wandb)
    plot_CL_metrics_from_logs(args.output_dir, "holdout", args.algorithm, exp_name, wandb)

    # Save the list of client trackers to use for future plots
    save_objects(client_trackers, os.path.join(args.output_dir, f"client_trackers_{exp_name}.pkl"))

    if args.is_novel_disease:
        metric_name = "holdout_per_category_acc_dict"
        log_dir = os.path.join(args.output_dir, "logs")
        common_acc_mean, common_acc_std, novel_acc_mean, novel_acc_std = (
            compute_novel_disease_results(log_dir, args.algorithm, metric_name)
        )
        print("----- Novel Disease Performance Summary -----")
        print("Results for last task:")
        print(f"Non-COVID LTR Acc.: {common_acc_mean[-1]:.2f} +- {common_acc_std[-1]:.1f}")
        print(f"COVID LTR Acc.: {novel_acc_mean[-1]:.2f} +- {novel_acc_std[-1]:.1f}")
        print("--------------------------------------------------")

    end_time = time.time()
    duration_seconds = end_time - start_time
    duration_hours = duration_seconds / 3600  # Convert seconds to hours
    print(f"Baseline experiment completed in {duration_hours:.2f} hours")
