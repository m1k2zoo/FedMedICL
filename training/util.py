import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import random
import numpy as np
from pathlib import Path
from model.resnet import ResNet
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from reporting.metrics import ClientTrackers

from model.models.swad import AveragedModel, LossValley


class FairFed:
    def __init__(self, client_dataset_list, device, beta=1):
        self.device = device
        self.beta = beta
        self.acc_global_t, self.acc_k_t = 0.0, torch.tensor(
            [0.0] * len(client_dataset_list), device=self.device
        )
        # self.F_global_t, self.F_k_t = None, None # Not applicable to FedMedICL
        self.weights_prev_t = self.init_weights(client_dataset_list)

    def init_weights(self, client_dataset_list):
        """
        Initializes the weights for each client based on the number of samples in their dataset.

        Parameters:
        - client_dataset_list (list of DatasetSplit): Each element is a dataset object for a client.

        Returns:
        - torch.Tensor: A tensor of the initial weights for each client.
        """
        # Calculate the number of samples for each client
        n_k = torch.tensor(
            [len(dataset.train_set) for dataset in client_dataset_list],
            dtype=torch.float,
            device=self.device,
        )
        # Calculate the total number of samples across all clients
        total_n = torch.sum(n_k)
        # Initialize weights for each client
        weights_0 = n_k / total_n

        return weights_0


class ClientAlgorithm:
    def __init__(self, model, device, is_swad=False):
        """
        Class to track data for each client.

        Args:
            is_swad (bool): Flag to indicate whether SWAD algorithm is enabled. Defaults to False.
        """

        # SWAD algorithm
        self.swad = None
        self.swad_early_stop = False
        self.device = device
        if is_swad:
            self.swad_n_converge = 3
            self.swad_n_tolerance = 6
            self.swad_tolerance_ratio = 0.05

            self.swad_step = 0
            # TODO: Try training with n_tolerance = swad_n_converge + swad_n_tolerance similar to MEDFAIR
            #  self.swad = LossValley(n_converge = swad_n_converge, n_tolerance = swad_n_converge + swad_n_tolerance,
            #    tolerance_ratio =swad_tolerance_ratio)
            self.swad = LossValley(
                n_converge=self.swad_n_converge,
                n_tolerance=self.swad_n_tolerance,
                tolerance_ratio=self.swad_tolerance_ratio,
            )
            self.swad_model = AveragedModel(model).to(device)
            self.swad_last_test_model = None

    def reset_SWAD(self, model):
        self.swad_early_stop = False
        self.swad = LossValley(
            n_converge=self.swad_n_converge,
            n_tolerance=self.swad_n_tolerance,
            tolerance_ratio=self.swad_tolerance_ratio,
        )
        self.swad_model = AveragedModel(model).to(self.device)
        self.swad_last_test_model = None


def mixup_data(x, y, device, alpha=1.0):
    """
    Applies Mixup augmentation to a batch of input data and labels.

    Parameters:
        x (Tensor): Batch of input data.
        y (Tensor): Batch of labels corresponding to `x`.
        device (str): The device where the computations will be performed.
        alpha (float, optional): Parameter of the Beta distribution used to generate the mixing coefficient. Default is 1.0.

    Returns:
        Tuple[Tensor, Tensor, Tensor, float]: A tuple containing the mixed inputs, original labels, shuffled labels, and mixing coefficient.
    """
    # Sample mixing coefficient from Beta distribution if alpha > 0, else set to 1
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    # Get batch size and shuffle indices for mixing
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    # Create mixed inputs and pair-wise labels
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def set_seed(seed):
    # Ensure reproducibility by setting the seed for random, numpy, and PyTorch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure reproducibility by eliminating non-determinism caused by inconsistent algorithm selection
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_args_pickle(args, output_dir):
    """
    Save the main args into args.output_dir.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        output_dir (str): Output directory to save the args.
    """

    # Create the output_dir directory if it does not exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    args_path = os.path.join(output_dir, "args.pkl")
    with open(args_path, "wb") as f:
        pickle.dump(args, f)


def setup(args):
    """
    Create the output directory if it doesn't exist, save the main args if distributed, and initialize wandb.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save the command-line arguments to a text file
    args_file_path = os.path.join(args.output_dir, "args.txt")
    with open(args_file_path, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # Save the main args as pickle, if distributed
    if args.use_distributed:
        save_args_pickle(args, args.output_dir)

    # Initialize wandb
    if args.use_wandb:
        import wandb

        # TODO: Pass an experiment name
        project_name = "fedmedical"
        if args.jt_iters > 0:
            project_name = "fedmedical_jt"
        wandb.init(project=project_name, config=args)

        # Define a custom x-axis for WandB metrics
        wandb.define_metric("train/*", step_metric="train/iteration")
        wandb.define_metric("val/*", step_metric="val/iteration")
        wandb.define_metric("test/*", step_metric="test/iteration")

    else:
        wandb = None

    return wandb


def initialize_models_and_optimizers(args, client_dataset_list, best_lr_dict):
    """
    Create num_clients copies of the model (only supports ResNet for now), initialize the criterion,
    optimizer, and ClientTrackers.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        client_dataset_list (list): List of client datasets.
        best_lr_dict (dict): Dictionary of best learning rates for each dataset.

    Returns:
        models (list): List of client models.
        optimizers (list): List of client optimizers.
    """
    # Create num_clients copies of the model (only supports ResNet for now)
    models = [
        ResNet(args.architecture, num_classes=client_dataset_list[i].num_classes).to(args.device)
        for i in range(args.num_clients)
    ]

    # Initialize the criterion, optimizer, and ClientTrackers
    # criterion = initialize_criterion(args.criterion)  # Could depend on the dataset task in the future
    best_lrs = [best_lr_dict[client_dataset_list[i].name] for i in range(args.num_clients)]
    optimizers = [
        initialize_optimizer(args.optimizer, models[i].parameters(), best_lrs[i], args.weight_decay)
        for i in range(args.num_clients)
    ]
    # schedulers = [initialize_scheduler(optimizers[client_id]) for client_id in range(args.num_clients)]

    return models, optimizers


def initialize_client_trackers(args, client_dataset_list, algorithm):
    """
    Create ClientTrackers.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        client_dataset_list (list): List of client datasets.
        algorithm (str): The name of the algorithm.

    Returns:
        client_trackers (list): List of client trackers.
    """
    client_trackers = [
        ClientTrackers(
            args.num_tasks,
            client_dataset.num_groups,
            client_dataset.client_id,
            client_dataset.num_classes,
            algorithm,
            args.use_macro_avg,
        )
        for client_dataset in client_dataset_list
    ]
    return client_trackers


def initialize_criterion(criterion):
    """
    Initialize the PyTorch criterion based on the provided string.

    Args:
        criterion (str): Loss function criterion.

    Returns:
        torch.nn.modules.loss._Loss: Initialized criterion.
    """
    if criterion == "mse":
        return nn.MSELoss()
    elif criterion == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid criterion specified!")


def initialize_optimizer(optimizer, model_parameters, lr=0.01, weight_decay=0):
    """
    Initialize the PyTorch optimizer based on the provided string.

    Args:
        optimizer (str): Optimizer for model parameters.
        model_parameters (iterable): Iterable of model parameters.
        lr (float): Learning rate to optimize with.
        weight_decay (float): Controls L2 regularization.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.
    """
    if optimizer == "sgd":
        return optim.SGD(model_parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == "adam":
        return optim.Adam(model_parameters, lr=lr)
    else:
        raise ValueError("Invalid optimizer specified!")


def initialize_scheduler(optimizer, mode="min", factor=0.5, patience=5, verbose=True, **kwargs):
    """
    Initialize the ReduceLROnPlateau scheduler for the given optimizer.

    Args:
        optimizer (torch.optim.Optimizer): Optimizer associated with the model parameters.
        mode (str): One of 'min' or 'max'. In 'min' mode, lr will be reduced when the quantity monitored has stopped decreasing;
                    in 'max' mode it will be reduced when the quantity monitored has stopped increasing.
        factor (float): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        patience (int): Number of evaluations with no improvement after which learning rate will be reduced.
        verbose (bool): If True, prints a message to stdout for each update.
        **kwargs: Additional arguments for ReduceLROnPlateau.

    Returns:
        ReduceLROnPlateau: Initialized learning rate scheduler.
    """
    return ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose, **kwargs
    )


def reset_learning_rate_and_scheduler(optimizers, schedulers, init_lr_dict):
    """
    Resets optimizers' learning rates to initial values and reinitializes schedulers for each client.

    Intended for use in continual learning or at the start of new tasks to ensure learning rates and schedulers are correctly reset.

    Args:
        optimizers (list): Optimizers for each client's model parameters.
        schedulers (list): Learning rate schedulers for each client, to be reinitialized.
        init_lr_dict (list): inital learning rates for each client.

    Returns:
        None: Updates `optimizers` and `schedulers` in-place.
    """
    num_clients = len(optimizers)
    for client_id in range(num_clients):
        # Reset learning rate for each client
        for param_group in optimizers[client_id].param_groups:
            param_group["lr"] = init_lr_dict[client_id]

        # Reinitialize the scheduler for each client's optimizer
        schedulers[client_id] = initialize_scheduler(optimizers[client_id])


def initialize_classifier_optimizer(model, optimizer):
    """
    Freezes the backbone layers of the model and initializes a new SGD optimizer
    for the output layer using parameters extracted from the last parameter group
    of the provided optimizer.

    Args:
        model (FoundationModel): The model containing the backbone and output layers.
        optimizer (Optimizer): The existing optimizer from which learning rate,
            weight decay, and momentum values will be extracted.

    Returns:
        Optimizer: A new SGD optimizer configured with the extracted parameters and
            targeting only the parameters of the model's output layer.

    Note:
        The method assumes the existence of 'lr' and optionally 'weight_decay' and
        'momentum' in the last parameter group of the provided optimizer. If
        'weight_decay' or 'momentum' are not found, default values of 0 are used.
    """
    # Freeze all parameters in the backbone layers
    for param in model.backbone_layers.parameters():
        param.requires_grad = False

    # Extract parameters from the last param group of the existing optimizer
    last_group = optimizer.param_groups[-1]
    lr = last_group["lr"]
    weight_decay = last_group.get("weight_decay", 0)  # Provide a default if not found
    momentum = last_group.get("momentum", 0.0)  # Provide a default if not found

    # Initialize a new SGD optimizer for the output layer
    new_optimizer = optim.SGD(
        model.output_layer.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    return new_optimizer


class InfiniteDataLoader:
    """This class allows to iterate the dataloader infinitely batch by batch.
    When there are no more batches the iterator is reset silently.
    This class allows to keep the memory of the state of the iterator hence its
    name.
    """

    def __init__(
        self,
        train_dataset,
        client_buffer,
        batch_size,
        num_workers,
        use_rehearsal,
        enable_balancing,
        imbalance_type="target",
    ):
        """This initialization takes a dataloader and creates an iterator object
        from it.

        Parameters
        - train_dataset (dataset.Dataset2D): The client-specific dataset for the current task.
        - client_buffer (ClientBuffer): The buffer for storing client-specific data from past tasks.
        - batch_size (int): The batch size for the DataLoader.
        - num_workers (int): The number of worker processes to use for data loading.
        - use_rehearsal (bool): Flag to indicate whether to enable data rehearsal
        - enable_balancing (bool): Flag to enable imbalance sampler for all DataLoaders.
        - imbalance_type (str): Determine whether to create balanced batches based on groups or targets.  Can be 'group' or 'target'

        """
        sampler = None
        self.buffer = client_buffer
        self.use_rehearsal = use_rehearsal
        if self.use_rehearsal:
            # if len(self.buffer) > 0:
            #     if batch_size % 2 != 0:
            #         raise ValueError(f"batch_size ({batch_size}) must be divisible by 2.")
            #     else:
            #         batch_size = batch_size//2

            # Buffer iterator
            if len(self.buffer.dataset) > 0:
                if enable_balancing:  # Using class/group balanced samplers
                    print("\n\nUsing a class-balanced sampler for the rehearsal dataloader!\n\n")
                    buffer_sampler = ImbalancedDatasetSampler(self.buffer.dataset)
                    self.buffer_dataloader = DataLoader(
                        self.buffer.dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        sampler=buffer_sampler,
                    )
                else:  # Using naive random sampler
                    self.buffer_dataloader = DataLoader(
                        self.buffer.dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=True,
                    )
                self.buffer_iterator = iter(self.buffer_dataloader)
            else:
                self.buffer_iterator = iter([])

        # Training iterator
        ### Code for the group resampling baseline
        if enable_balancing:
            if imbalance_type == "group":
                from torch.utils.data.sampler import WeightedRandomSampler

                print("\n\nUsing a group-balanced sampler for the training dataloader!\n\n")
                attributes = train_dataset.get_attribute_groups(imbalance_type)
                group_distribution = {}
                for group in attributes:
                    group_name = group.attribute_group
                    group_index, group_size = train_dataset.get_groups().index(group_name), len(
                        group
                    )
                    group_distribution[group_index] = group_size

                weights = [1.0 / group_distribution[sensitive] for _, _, sensitive in train_dataset]
                weights = torch.DoubleTensor(weights)
                sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

            else:
                print("\n\nUsing a class-balanced sampler for the training dataloader!\n\n")
                sampler = ImbalancedDatasetSampler(train_dataset)  # class-balanced loading
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            shuffle=False,
        )
        self.train_iterator = iter(self.train_dataloader)
        # TEST: group-based balanced dataloader
        # from collections import Counter
        # for i in range(10):
        #     # Sample a batch from the iterator
        #     images, targets, sensitive_values = next(self.train_iterator)

        #     # Convert sensitive_values tensor to a list for counting
        #     sensitive_values_list = sensitive_values.tolist()

        #     # Use Counter to count occurrences of each sensitive value
        #     sensitive_counts = Counter(sensitive_values_list)

        #     # Print the counts for each sensitive value in the batch
        #     print(sensitive_counts)

        # print("DONE!")

    def _reset_iterator(self, iterator_type):
        if iterator_type == "train":
            self.train_iterator = iter(self.train_dataloader)
            return self.train_iterator
        elif iterator_type == "buffer":
            self.buffer_iterator = iter(self.buffer_dataloader)
            return self.buffer_iterator
        else:
            raise ValueError("Invalid type specified. Expected 'train' or 'buffer'.")

    def __len__(self):
        if self.use_rehearsal:
            return len(self.train_iterator) + len(self.buffer_iterator)
        else:
            return len(self.train_iterator)

    def get_next_batch(self, iterator_type):
        """
        Get the next batch from the specified iterator type or reset it if needed.

        Parameters:
        - iterator_type (str): Specifies whether the iterator is "train" or "buffer".

        Returns:
        tuple: A batch from the iterator, including the input, output, and sensitive attribute.
        """

        if iterator_type == "train":
            iterator = self.train_iterator
        elif iterator_type == "buffer":
            iterator = self.buffer_iterator
        else:
            raise ValueError("Invalid type specified. Expected 'train' or 'buffer'.")

        try:
            X, y, attribute = next(iterator)
            batch = (X, y, attribute)
        except StopIteration:
            iterator = self._reset_iterator(iterator_type)

            # Try again to get the batch
            X, y, attribute = next(iterator)
            batch = (X, y, attribute)
        return batch

    def get_samples(self):
        """This method generates the next batch from the iterator or resets it
        if needed. It can be called an infinite amount of times.

        Returns
        -------
        tuple
            a batch from the iterator, including the input, output, and sensitive attribute
        """
        train_batch = self.get_next_batch(iterator_type="train")

        if len(self.buffer) == 0:
            # Get train batches only since buffer is empty
            X, y, attribute = train_batch
        else:
            if not self.use_rehearsal:
                raise RuntimeError(
                    "Buffer is not empty, but 'use_rehearsal' is False. This should never happen."
                )

            # Concatenate the train and buffer batches
            buffer_batch = self.get_next_batch(iterator_type="buffer")

            X = torch.cat((train_batch[0], buffer_batch[0]), dim=0)  # Input data
            y = torch.cat((train_batch[1], buffer_batch[1]), dim=0)  # Output data
            attribute = torch.cat((train_batch[2], buffer_batch[2]), dim=0)  # Sensitive attribute

        return X, y, attribute
