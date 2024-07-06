import argparse
import json
import os


def parse_args():
    """
    Parses command-line arguments for configuring the training environment of distributed client models.

    This function sets up the environment for training machine learning models across multiple datasets,
    handling various configurations such as model architecture, optimizer settings, and federated learning parameters.

    Returns:
        argparse.Namespace: An object that contains all the command-line arguments. The attributes of this object
        are accessible as direct named attributes of a namespace.
    """
    parser = argparse.ArgumentParser(description="Training Task")
    model_choices = ["resnet18", "resnet34", "resnet50"]

    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/datasets.json",
        help="Path to dataset configuration file",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        default="resnet18",
        choices=model_choices,
        help="Name of the model architecture (e.g. resnet34)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        choices=["mse", "cross_entropy"],
        help="Loss function criterion",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer for model parameters",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="fitzpatrick17k,HAM10000,PAPILA,OL3I",
        help="Comma-separated list of dataset names",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the training on")
    parser.add_argument(
        "--num_iters",
        type=int,
        default=5,
        help="Number of training iterations before federated aggregation",
    )
    parser.add_argument(
        "--num_rounds", type=int, default=150, help="Number of federated rounds per task"
    )
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for training data loaderss"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=10, help="Batch size for evaluation data loaders"
    )
    parser.add_argument(
        "--num_workers", type=int, default=3, help="Number of workers for training data loaders"
    )
    parser.add_argument(
        "--eval_num_workers",
        type=int,
        default=6,
        help="Number of workers for evaluation data loaders",
    )
    parser.add_argument("--num_clients", type=int, default=10, help="Number of clients")
    parser.add_argument(
        "--num_tasks", type=int, default=4, help="Number of continual learning tasks"
    )
    parser.add_argument(
        "--training_log_frequency",
        type=int,
        default=100,
        help="The frequency, in training iterations, at which to log training results",
    )
    parser.add_argument(
        "--num_evaluations",
        type=int,
        default=3,
        help="The number of times to evaluate a model during its task training process",
    )
    parser.add_argument(
        "--use_wandb", action="store_true", default=False, help="Use wandb for logging"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Path for saving client files and results"
    )
    parser.add_argument(
        "--seed", type=int, default=13, help="The seed value to reproduce the randomness"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="The weight decay used by the optimizer"
    )

    # [Deprecated: This flag is no yet fully supported.]
    # parser.add_argument(
    #     "--use_distributed",
    #     action="store_true",
    #     default=False,
    #     help="Perform training in a distributed computation environment.",
    # )
    parser.add_argument(
        "--is_imbalanced",
        action="store_true",
        default=False,
        help="Perform training with imbalanced client datasets.",
    )
    parser.add_argument(
        "--imbalance_ratios", type=str, help="JSON of the imbalance ratios of clients."
    )
    parser.add_argument(
        "--imbalance_type",
        type=str,
        default="group",
        help="Determine whether to create imbalanced clients based on groups or targets.",
    )
    parser.add_argument(
        "--task_split_type",
        type=str,
        default="Naive",
        help="Type of split to use for creating tasks.",
    )
    parser.add_argument(
        "--task_split_ratios", type=str, default=None, help="List of task split ratios."
    )
    parser.add_argument(
        "--skip_lr_search",
        action="store_true",
        default=False,
        help="Skip searching for learing rate, use a predefined value instead.",
    )
    parser.add_argument(
        "--is_novel_disease",
        default=False,
        action="store_true",
        help="Analysis experiment in which some clients have access to a novel disease.",
    )
    parser.add_argument(
        "--apply_normalizer",
        default=False,
        action="store_true",
        help="Apply a Normalizer to models during federated aggregation.",
    )
    parser.add_argument(
        "--use_macro_avg",
        default=False,
        action="store_true",
        help="Flag to indicate if macro-averaged (per-category) accuracy should be computed.",
    )
    parser.add_argument(
        "--average_all_layers",
        action="store_true",
        default=False,
        help="Average all layers during federated aggregation, not just the backbone.",
    )
    parser.add_argument(
        "--jt_iters",
        type=int,
        default=0,
        help="Number of training iterations for the joint-training baseline.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="ERM",
        help="Specify if a baseline training algorithm from another benchmark is used.",
    )
    # The lr_dict_path should indicate the architecture name, e.g. 'ResNet18'
    parser.add_argument(
        "--lr_dict_path",
        type=str,
        default="configs/best_lr.json",
        help="Path to a saved dictionary of dataset best learning rates",
    )

    # Arguments for FedDC (daisy chaining)
    parser.add_argument(
        "--feddc_daisy",
        type=int,
        default=1,
        help="Number of federated rounds before daisy chaining",
    )
    # parser.add_argument('--feddc_aggregate', type=int, default=5, help='Number of federated rounds before federated aggregation')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Since this flag is not yet supported, we set it to false
    args.use_distributed = False

    # TODO: consider removing this temproary override
    args.feddc_aggregate = args.num_iters

    # Default values for "ERM" (LT) algorithm
    args.use_fl = False
    args.dataloader_balancing = False
    args.use_rehearsal = False
    if args.algorithm in ["CB", "resampling"]:
        args.use_fl = True
        args.dataloader_balancing = True
    elif args.algorithm in ["CRT", "mixup", "SWAD", "fedavg", "feddc", "fairfed"]:
        args.use_fl = True
    elif args.algorithm == "ER":
        args.use_fl = True
        args.use_rehearsal = True
    elif args.algorithm == "FACTR":
        args.use_fl = True
        args.dataloader_balancing = True
        args.use_rehearsal = True

    dataset_name = ""
    if "," not in args.datasets:
        dataset_name = f"{args.datasets}"

    if args.jt_iters > 0:
        exp_folder_name = f"{dataset_name}_JT_{args.jt_iters}I"
    else:
        exp_folder_name = f"{dataset_name}_{args.num_tasks}T_{args.num_clients}C_{args.num_rounds}R_{args.num_iters}I_{args.algorithm}"

        if args.num_tasks > 1:
            exp_folder_name += f"_{args.task_split_type}"

        if args.is_novel_disease:
            exp_folder_name += f"_novel"

    output_dir = os.path.join(args.output_dir, exp_folder_name)
    # Check if the output directory already exists
    if os.path.exists(output_dir):
        suffix = 1
        while os.path.exists(f"{output_dir}_copy{suffix}"):
            suffix += 1
        output_dir = f"{output_dir}_copy{suffix}"
    args.output_dir = output_dir

    args.datasets = args.datasets.split(",")

    # Convert string to dictionary
    if args.imbalance_ratios:
        args.imbalance_ratios = json.loads(args.imbalance_ratios)

    total_iterations = args.num_iters * args.num_rounds
    if args.training_log_frequency > total_iterations:
        print(
            "The value of 'training_log_frequency' exceeds the total number of iterations. It will be adjusted to be equal to the total iterations."
        )
        args.training_log_frequency = total_iterations

    if args.task_split_ratios is not None:
        import ast

        args.task_split_ratios = ast.literal_eval(args.task_split_ratios)

    return args


def parse_client_args():
    """
    Parser for training a single client models on a single task.
    """
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Client Training")

    # Add the required command-line arguments
    parser.add_argument("--task_id", type=int, help="The continual learning task ID")
    parser.add_argument("--client_id", type=int, help="The client index")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="client_files",
        help="Path for saving client files and results",
    )

    # Parse the command-line arguments
    args = parser.parse_args()
    return args
