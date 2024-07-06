import os
import pickle
import subprocess
import time
import torch
from model.resnet import ResNet

from training.util import initialize_optimizer

from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import DataLoader
from training.util import InfiniteDataLoader


def save_dataloader_state(round, args):
    """
    Save the state of the dataloader for all clients to a file.

    The function saves the number of sampled batches and the current round for each client.

    Parameters:
    - round (int): The current round number in the training process.
    - args (argparse.Namespace): Parsed command-line arguments.

    This function iterates over each client, creating a subdirectory for their specific data if not already present,
    and writes their dataloader state to a pickle file. This state includes the total number of batches that have
    been processed up to the current round, which is essential for resuming training in a consistent state.

    """
    for i in range(args.num_clients):
        client_subdir = os.path.join(args.output_dir, f"client_{i}")
        os.makedirs(client_subdir, exist_ok=True)
        num_sampled_batches = round * args.num_iters
        # Save the state
        state = {
            "num_sampled_batches": num_sampled_batches,
            "round": round,  # save the round too, for sanity check during loading
        }
        dataloader_path = os.path.join(client_subdir, "dataloader_state.pkl")
        with open(dataloader_path, "wb") as f:
            pickle.dump(state, f)


def load_dataloader_state(dataset, output_dir, client_id, args):
    """
    Load the state of the dataloader for a specific client from a file.

    The function creates a InfiniteDataLoader instance and sets its iterator to the state
    saved in the file, making it ready for further sampling.

    Parameters:
    - dataset (Dataset): The dataset object containing the train_set.
    - output_dir (str): The directory from where the dataloader states should be loaded.
    - client_id (int): The ID of the client for which the state should be loaded.
    - args (argparse.Namespace): Parsed command-line arguments.

    Returns:
    InfiniteDataLoader: The dataloader with its state set to the saved state.

    This function sets up a DataLoader with an ImbalancedDatasetSampler, based on the training set from the dataset,
    and adjusts its iteration state to match the point saved in a previous training round. This ensures that the
    training process can continue seamlessly after a restart or pause.
    """
    sampler = ImbalancedDatasetSampler(dataset.train_set)
    train_loader = DataLoader(
        dataset.train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        shuffle=False,
    )
    client_infinite_loader = InfiniteDataLoader(train_loader)

    client_subdir = os.path.join(output_dir, f"client_{client_id}")
    dataloader_path = os.path.join(client_subdir, "dataloader_state.pkl")

    with open(dataloader_path, "rb") as f:
        state = pickle.load(f)

    num_sampled_batches = state["num_sampled_batches"]
    for _ in range(num_sampled_batches):
        client_infinite_loader.get_samples()

    return client_infinite_loader


def test_dataloader_consistency(dataset, args, client_id, num_rounds_to_test=5):
    """
    Test the consistency of the dataloader state after saving and reloading.

    Parameters:
    - dataset: The dataset object containing the train_set.
    - args: Arguments containing batch_size, num_workers, etc.
    - client_id: ID of the client for which state should be tested.
    - num_rounds_to_test: Number of rounds to test consistency for. Default is 5.

    Returns:
    - bool: True if consistent, False otherwise.
    """

    output_dir = args.output_dir

    # Construct the original dataloader
    sampler = ImbalancedDatasetSampler(dataset.train_set)
    train_loader = DataLoader(
        dataset.train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sampler=sampler,
        shuffle=False,
    )
    original_loader = InfiniteDataLoader(train_loader)

    # Consume initial batches from the original dataloader
    _ = [original_loader.get_samples() for _ in range(num_rounds_to_test * args.num_iters)]

    # Save the dataloader state
    save_dataloader_state(num_rounds_to_test, args)

    # Load the dataloader state
    loaded_loader = load_dataloader_state(dataset, output_dir, client_id, args)

    # Consume the same number of batches from the loaded dataloader
    original_batches = [
        original_loader.get_samples() for _ in range(num_rounds_to_test * args.num_iters)
    ]
    loaded_batches = [
        loaded_loader.get_samples() for _ in range(num_rounds_to_test * args.num_iters)
    ]

    # Compare the batches
    for original_batch, loaded_batch in zip(original_batches, loaded_batches):
        if (
            not torch.all(original_batch[0].eq(loaded_batch[0]))
            or not torch.all(original_batch[1].eq(loaded_batch[1]))
            or not torch.all(original_batch[2].eq(loaded_batch[2]))
        ):
            return False

    return True


def save_datasets_and_trackers(client_dataset_list, client_trackers, client_subdirs):
    """
    Save each client's dataset and client tracker as pickle objects.

    Args:
        client_dataset_list (list): List of client datasets.
        client_trackers (list): List of client trackers.
        client_subdirs (list): List of subdirectories for each client's dataset.
    """
    for i, (client_dataset, client_tracker) in enumerate(zip(client_dataset_list, client_trackers)):
        dataset_path = os.path.join(client_subdirs[i], "client_dataset.pkl")
        with open(dataset_path, "wb") as f:
            pickle.dump(client_dataset, f)

        tracker_path = os.path.join(client_subdirs[i], "client_tracker.pkl")
        with open(tracker_path, "wb") as f:
            pickle.dump(client_tracker, f)


def save_models_and_optimizers(models, optimizers, client_subdirs):
    """
    Save each model and optimizer together in one dictionary.

    Args:
        models (list): List of models.
        optimizers (list): List of optimizers.
        client_subdirs (list): List of subdirectories for each client's dataset.
    """
    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        model_path = os.path.join(client_subdirs[i], "model_optimizer.pth")
        torch.save(state_dict, model_path)


def load_client_data(output_dir, client_id):
    """
    Load and return the corresponding args, model, optimizer, dataset, and tracker for a given client.

    Args:
        output_dir (str): Output directory where the client data is stored.
        client_id (int): ID of the client.

    Returns:
        args (argparse.Namespace): Parsed command-line arguments.
        model (torch.nn.Module): Loaded model.
        optimizer (torch.optim.Optimizer): Loaded optimizer.
        dataset (object): Loaded dataset.
        tracker (object): Loaded tracker.
    """
    client_dir = os.path.join(output_dir, f"client_{client_id}")

    # Load args
    args_path = os.path.join(output_dir, "args.pkl")
    with open(args_path, "rb") as f:
        args = pickle.load(f)

    # Load dataset
    dataset_path = os.path.join(client_dir, "client_dataset.pkl")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Load model and optimizer
    model_path = os.path.join(client_dir, "model_optimizer.pth")
    # Specify the map location to avoid device errors
    checkpoint = torch.load(model_path, map_location=args.device)
    model = ResNet(args.architecture, num_classes=dataset.num_classes)
    model.load_state_dict(checkpoint["model"])
    optimizer = initialize_optimizer(args.optimizer, model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer"])

    # Load tracker
    tracker_path = os.path.join(client_dir, "client_tracker.pkl")
    with open(tracker_path, "rb") as f:
        tracker = pickle.load(f)

    return args, model, optimizer, dataset, tracker


def load_models(output_dir):
    """
    Load and return a list of models for all clients in the given output directory.

    Args:
        output_dir (str): Output directory where the client data is stored.

    Returns:
        models (list): List of loaded models for all clients.
    """
    models = []
    client_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    # Load args
    args_path = os.path.join(output_dir, "args.pkl")
    with open(args_path, "rb") as f:
        args = pickle.load(f)

    for client_dir in client_dirs:
        model_path = os.path.join(output_dir, client_dir, "model_optimizer.pth")
        checkpoint = torch.load(model_path, map_location=args.device)

        # Load dataset (needed to know the number of classes)
        dataset_path = os.path.join(output_dir, client_dir, "client_dataset.pkl")
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        model = ResNet(args.architecture, num_classes=dataset.num_classes)
        model.load_state_dict(checkpoint["model"])
        models.append(model)

    return models


def load_client_trackers(output_dir):
    """
    Load and return a list of client trackers for all clients in the given output directory.

    Args:
        output_dir (str): Output directory where the client data is stored.

    Returns:
        client_trackers (list): List of loaded client trackers for all clients.
    """
    client_trackers = []
    client_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]

    for client_dir in client_dirs:
        tracker_path = os.path.join(output_dir, client_dir, "client_tracker.pkl")
        with open(tracker_path, "rb") as f:
            tracker = pickle.load(f)
        client_trackers.append(tracker)

    return client_trackers


def save_objects(object_list, filename):
    """
    Save a list of Python objects to a file using pickle.

    Args:
        object_list (list): A list of Python objects.
        filename (str): The name of the file to save the data.

    Raises:
        IOError: If there is an error while saving the file.
    """
    try:
        with open(filename, "wb") as file:
            pickle.dump(object_list, file)
    except IOError as e:
        print(f"Error occurred while saving the file: {e}")
        raise e


def load_objects(filename):
    """
    Load a list of Python objects from a file using pickle.

    Args:
        filename (str): The name of the file to load the data from.

    Returns:
        list: A list of Python objects.

    Raises:
        IOError: If there is an error while loading the file.
    """
    try:
        with open(filename, "rb") as file:
            object_list = pickle.load(file)
            return object_list
    except IOError as e:
        print(f"Error occurred while loading the file: {e}")
        raise e


def run_client_job(task_id, client_id, output_dir):
    """
    Submits a job for training a client on a specific task to a computing cluster.

    Args:
        task_id (int): The ID of the task.
        client_id (int): The ID of the client.
        output_dir (str): The path to the output directory.

    Returns:
        subprocess.Popen: A subprocess representing the running job.
    """

    # This function is intended to submit a job for training a client on a specific task
    # to a computing cluster using a job scheduler (e.g., Slurm with sbatch).
    # The implementation details will depend on your computing environment and job
    # submission setup. Below is a placeholder implementation that needs adaptation
    # to your specific setup.
    raise NotImplementedError("This function is a placeholder and needs implementation.")

    # Create a directory specific to the client within the output directory
    client_dir = f"{output_dir}/client_{client_id}"

    # Construct the output and error file paths for the job
    job_name = f"T{task_id}_C{client_id}"
    job_output = f"{client_dir}/task{task_id}.out"
    job_error = f"{client_dir}/task{task_id}.err"

    # Construct the command for job submission, including the necessary arguments
    cmd = [
        "sbatch",
        "--job-name",
        job_name,
        "--output",
        job_output,
        "--error",
        job_error,
        "scripts/run_client.sh",
        "--task_id",
        str(task_id),
        "--client_id",
        str(client_id),
        "--output_dir",
        output_dir,
    ]

    # Submit the job
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Return the subprocess representing the running job
    return p


def is_job_successful(job_id):
    """
    Check if a job with the given job ID ended successfully.

    Args:
        job_id (int): The ID of the job to check.

    Returns:
        bool: True if the job ended successfully, False otherwise.
    """
    cmd = ["sacct", "-j", str(job_id), "--format", "State", "--noheader"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    state_lines = stdout.decode().strip().split("\n")
    if len(state_lines) > 0:
        state = state_lines[0].strip()
        return state == "COMPLETED"
    return False


def is_job_done(job_id):
    """
    Check if a job with the given job ID is no longer in the queue.

    Args:
        job_id (int): The ID of the job to check.

    Returns:
        bool: True if the job is no longer in the queue, False otherwise.
    """
    cmd = ["squeue", "-j", str(job_id)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return str(job_id) not in stdout.decode()


def wait_for_job_completion(job_id, client_id, task_id):
    """
    Wait for a job with the given job ID to complete.

    Args:
        job_id (int): The ID of the job to wait for.
        client_id (int): The ID of the client associated with the job.
        task_id (int): The ID of the task associated with the job.
    """
    while not is_job_done(job_id):
        time.sleep(2)  # Sleep for 2 seconds before checking again

    if not is_job_successful(job_id):
        raise RuntimeError(f"Job ID {job_id} for task {task_id} and client {client_id} failed")

    print(f"Job ID {job_id} for task {task_id} and client {client_id} is completed")
