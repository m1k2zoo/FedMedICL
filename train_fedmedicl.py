import warnings
from training.util import set_seed, setup
from training.tuning import find_best_lr_for_datasets
from dataset.util.dataset_helper import prepare_datasets, prepare_clients_tasks
from helpers.argparser import parse_args
from training.experiment_runner import run_joint_baseline, run_main_experiment
from helpers.client_management import *
import copy


def adjust_num_tasks(args, datasets_list):
    """
    Adjust the 'num_tasks' parameter based on dataset characteristics and task split type.
    """
    if len(datasets_list) == 1 and args.task_split_type == "class_incremental":
        # TEMPORARY, TODO: consider removing later
        warnings.warn("Overwriting 'num_tasks' based on number of targets in the dataset.")
        args.num_tasks = len(datasets_list[0].targets_set)

    elif len(datasets_list) == 1 and args.task_split_type == "group_incremental":
        fine_grained_groups = [
            group.attribute_group[0]
            for group in datasets_list[0].train_set.get_attribute_groups("fine_grained_group")
        ]
        args.num_tasks = len(fine_grained_groups)
        if datasets_list[0].train_set.tasks_sensitive_name == "Age_multi":
            # Since we're combining age groups 0 (0-20 years old) and 1 (20-40 years old) into a single task for "Age_multi" datasets,
            # we subtract 1 from the number of tasks to account for this consolidation.
            if all(item in fine_grained_groups for item in [0, 1]):
                args.num_tasks -= 1
        warnings.warn(
            "Overwriting 'num_tasks' based on number of fine-grained groups in the dataset."
        )


def main(args):
    """
    Main function for training client models on distributed datasets.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    set_seed(args.seed)

    # Create the output directory if it doesn't exist, save the main args if distributed, and initialize wandb
    wandb = setup(args)

    # Load the dataset and distribute the dataset to clients
    datasets_list, client_dataset_list = prepare_datasets(
        args.datasets,
        args.num_clients,
        args.seed,
        args.config_path,
        args.output_dir,
        args.task_split_type,
        args.is_imbalanced,
        imbalance_type=args.imbalance_type,
        imbalance_ratios=args.imbalance_ratios,
        is_novel_disease=args.is_novel_disease,
    )
    # If args.is_novel_disease, each client has an attribute flag novel_disease

    adjust_num_tasks(args, datasets_list)

    # Perform LR hyperparameter search
    if args.lr_dict_path == "None":
        args.lr_dict_path = None
    best_lr_dict = find_best_lr_for_datasets(
        datasets_list,
        args.architecture,
        args.criterion,
        args.optimizer,
        args.device,
        args.batch_size,
        args.eval_batch_size,
        args.seed,
        args.output_dir,
        lr_dict_path=args.lr_dict_path,
        skip_lr_search=args.skip_lr_search,
    )

    if len(datasets_list) == 1 and args.jt_iters > 0:
        # Consider whether the JT training baseline is needed if args.num_tasks > 1
        jt_dataset = copy.deepcopy(datasets_list)  # deepcopy to keep the original dataset intact
        # Split client datasets into tasks
        if args.num_tasks > 1:
            jt_dataset = prepare_clients_tasks(
                datasets_list,
                jt_dataset,
                args.num_tasks,
                args.seed,
                args.config_path,
                args.output_dir,
                task_split_type=args.task_split_type,
                custom_task_split_ratios=args.task_split_ratios,
                is_joint_training=True,
            )
        print(
            "==================== Training Upper Bound Baseline (joint training) ===================="
        )
        run_joint_baseline(args, jt_dataset, best_lr_dict, wandb)

    else:
        # Split client datasets into tasks for local and federated training
        if args.num_tasks > 1:
            client_dataset_list = prepare_clients_tasks(
                datasets_list,
                client_dataset_list,
                args.num_tasks,
                args.seed,
                args.config_path,
                args.output_dir,
                task_split_type=args.task_split_type,
                custom_task_split_ratios=args.task_split_ratios,
                is_joint_training=False,
                is_novel_disease=args.is_novel_disease,
            )

        if args.use_fl:
            fl_status = "enabled"
        else:
            fl_status = "disabled"
        print("==================== Main Experiment ====================")
        print(f"Number of tasks: {args.num_tasks}, Federated Learning is {fl_status}")
        run_main_experiment(args, client_dataset_list, best_lr_dict, wandb)

    print("---- THE END ----")


if __name__ == "__main__":
    """
    Command-line arguments:
        --config_path (str): Path to dataset configuration file.
        --architecture (str): Name of the model architecture (e.g. resnet34).
        --criterion (str): Loss function criterion.
        --optimizer (str): Optimizer for model parameters.
        --datasets (str): Comma-separated list of dataset names.
        --device (str): Device to run the training on.
        --num_iters (int): Number of training iterations before federated aggregation.
        --num_rounds (int): Number of federated rounds per task.
        --batch_size (int): Batch size for training data loaders.
        --eval_batch_size (int): Batch size for evaluation data loaders.
        --num_clients (int): Number of clients.
        --num_tasks (int): Number of continual learning tasks.
        --training_log_frequency (int): The frequency, in training iterations, at which to log training results.
        --num_evaluations (int): The number of times to evaluate a model during its task training process.
        --use_wandb (bool): Use wandb for logging.
        --use_fl (bool): Use federated learning.
        --apply_normalizer (bool): Apply a Normalizer to models during federated aggregation.
        --output_dir (str): Path for saving client files and results.
        --seed (int): The seed value to reproduce the randomness.
        --use_distributed (boolean): [Deprecated: This flag is no yet fully supported.] Perform training in a distributed computation environment.
        --is_imbalanced (boolean): Perform training with imbalanced client datasets.
        --imbalance_ratios (str): JSON of the imbalance type of clients.
        --imbalance_type (str): Determine whether to create imbalanced clients based on groups or targets.
        --lr_dict_path (str): Path to a saved dictionary of dataset names as keys and best learning rates as values.
        --skip_lr_search (boolean): Skip searching for learing rate, use a predefined value instead.
        --use_macro_avg (boolean): Flag to indicate if macro-averaged (per-category) accuracy should be computed.
        --average_all_layers (boolean): Average all layers during federated aggregation, not just the backbone.
        --algorithm (str): Specify if a baseline training algorithm from another benchmark is used.
    """
    args = parse_args()

    # Print the arguments
    print("The experiment arguments:")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")

    main(args)
