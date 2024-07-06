import os
import json
import torch
from tqdm import tqdm
from model.resnet import ResNet
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler

from training.util import set_seed, initialize_criterion, initialize_optimizer


def find_best_lr_for_datasets(
    dataset_list,
    architecture,
    criterion_name,
    optimizer_name,
    device,
    batch_size,
    eval_batch_size,
    seed,
    output_dir,
    lr_dict_path=None,
    num_workers=6,
    num_epochs=10,
    skip_lr_search=False,
):
    """
    Finds the best learning rate for a given PyTorch model and a list of datasets.

    Parameters:
        dataset_list (list): A list of DatasetSplit datasets.
        architecture (str): Name of the model architecture (e.g. resnet34).
        criterion_name (str): The loss function name to use.
        optimizer_name (str): The optimizer name to use.
        device (str): device to use for training.
        batch_size (int): batch size for training data loaders.
        eval_batch_size (int): batch size for evaluation data loaders.
        seed (int): The seed value to reproduce the randomness.
        output_dir (str): The default output directory to save the best_lr_dict when lr_dict_path is None.
        lr_dict_path (str): Path to a saved dictionary of dataset names as keys and best learning rates as values.
                    If None, the best_lr_dict will be saved to the default output directory using output_dir.
                    If both lr_dict_path and output_dir are None, a ValueError will be raised.
        num_workers (int): The number of worker processes to use for loading data (default: 6).
        num_epochs (int): Number of epochs to train for (default: 10).
        skip_lr_search (boolean): A flag to skip learning rate search.

    Returns:
        A dictionary of dataset names as keys and best learning rates as values.
    """
    if skip_lr_search:
        temp_lr_dict = {}
        for dataset in dataset_list:
            dataset_name = dataset.name
            temp_lr_dict[dataset_name] = 0.001
        return temp_lr_dict

    best_lr_dict = {}
    # Load existing learning rate dictionary if provided
    if lr_dict_path is not None and os.path.isfile(lr_dict_path):
        with open(lr_dict_path, "r") as f:
            best_lr_dict = json.load(f)

    # Ensure the key for the current batch size exists
    batch_key = f"batch_size_{batch_size}"
    if batch_key not in best_lr_dict:
        best_lr_dict[batch_key] = {}

    for dataset in dataset_list:
        dataset_name = dataset.name
        print(f"LR Search for the dataset: {dataset_name}")

        # Check if the dataset already exists in the dictionary
        if dataset_name in best_lr_dict[batch_key]:
            print(
                f"Learning rate for {dataset_name} already exists in the dictionary. Skipping search.\n"
            )
            continue

        best_lr = find_best_lr(
            architecture,
            dataset,
            criterion_name,
            optimizer_name,
            device,
            batch_size,
            eval_batch_size,
            num_workers,
            num_epochs,
            seed,
        )

        print(f"Best learning rate for {dataset_name} is {best_lr}\n")
        best_lr_dict[batch_key][dataset_name] = best_lr

    # Save the updated learning rate dictionary if a path is provided
    if lr_dict_path is None:
        if output_dir is None:
            raise ValueError("Either lr_dict_path or output_dir should be provided.")
        os.makedirs(output_dir, exist_ok=True)
        lr_dict_path = os.path.join(output_dir, "best_lr.json")

    with open(lr_dict_path, "w") as f:
        json.dump(best_lr_dict, f, indent=4)

    return best_lr_dict[batch_key]


def find_best_lr(
    architecture,
    dataset,
    criterion_name,
    optimizer_name,
    device,
    batch_size,
    eval_batch_size,
    num_workers,
    num_epochs,
    seed,
):
    """
    Finds the best learning rate for a given PyTorch model and dataset.

    Parameters:
        architecture (str): Name of the model architecture (e.g. resnet34).
        dataset (DatasetSplit): A DatasetSplit instance containing the training and validation sets.
        criterion_name (str): The loss function name to use.
        optimizer_name (str): The optimizer name to use.
        device (str): device to use for training.
        batch_size (int): batch size for training data loaders.
        eval_batch_size (int): batch size for evaluation data loaders.
        num_workers (int): The number of worker processes to use for loading data.
        num_epochs (int): Number of epochs to train for.
        seed (int): The seed value to reproduce the randomness.

    Returns:
        The best learning rate found.
    """
    # Define dataloaders
    sampler = ImbalancedDatasetSampler(dataset.train_set)
    train_dataloader = DataLoader(
        dataset.train_set,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False,
    )
    val_dataloader = DataLoader(
        dataset.val_set, batch_size=eval_batch_size, num_workers=num_workers, shuffle=False
    )

    # Define loss function
    criterion = initialize_criterion(criterion_name)

    # Define learning rates to try
    lrs = [0.005, 0.001, 0.0005, 0.0001]
    print(f"Learning rates to try: {lrs}")
    # Track losses for each learning rate
    losses = []

    # Train and evaluate model with each learning rate
    for lr in tqdm(lrs, desc=f"Searching for best LR"):
        # Initialize a new model with the same parameters as the original model
        set_seed(seed)
        model = ResNet(architecture, num_classes=dataset.num_classes)
        model.to(device)

        # Define optimizer with current learning rate
        optimizer = initialize_optimizer(optimizer_name, model.parameters(), lr)

        # Train model for specified number of epochs
        for _ in tqdm(range(num_epochs), leave=False, desc=f"Epochs"):
            model.train()  # set model to training mode

            for inputs, targets, attributes in train_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Evaluate model on validation set
            model.eval()  # set model to evaluation mode
            with torch.no_grad():
                val_losses = []
                for val_inputs, val_targets, val_attributes in val_dataloader:
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)

                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_targets)

                    val_losses.append(val_loss.item())

                # Calculate average validation loss over all batches
                avg_val_loss = sum(val_losses) / len(val_losses)

        # Append the average validation loss
        losses.append(avg_val_loss)

    # Find best learning rate based on minimum validation loss
    min_loss_index = losses.index(min(losses))
    best_lr = lrs[min_loss_index]

    return best_lr
