from tqdm import tqdm
import copy
import torch
from torch.utils.data import DataLoader

from reporting.logging import log_training_results, log_evaluation_results, log_holdout_results
from training.util import initialize_criterion
from model.models.swad import AveragedModel, update_bn


def train_client(
    task_id,
    client_id,
    iteration_counter,
    model,
    optimizer,
    client_dataset,
    client_algorithm,
    metric_tracker,
    train_loader,
    args,
    wandb,
    jt_scheduler=None,
):
    """
    Train a client model on its corresponding dataset for a specific task.

    Args:
        task_id (int): ID of the task.
        client_id (int): ID of the client.
        iteration_counter (int): the number of completed training iterations
        model (torch.nn.Module): Client model.
        optimizer (torch.optim.Optimizer): Client optimizer.
        client_dataset (object): Client dataset.
        client_algorithm (object): Client algorithm.
        metric_tracker (ClientTrackers): Client tracker.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        args (argparse.Namespace): Parsed command-line arguments.
        wandb (wandb.run, optional): wandb run object for logging.
        jt_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): A learning rate scheduler for joint training.

    Returns:
        model (torch.nn.Module): Trained client model.
        optimizer (torch.optim.Optimizer): Updated client optimizer.
        client_dataset (object): Client dataset.
        metric_tracker (ClientTrackers): Client tracker.
    """
    model.train()
    model = model.to(args.device)
    criterion = initialize_criterion(args.criterion)

    # best_val_accuracy = 0.0
    # best_model = None

    for _ in range(args.num_iters):
        # Fetching a batch from InfiniteDataLoader
        inputs, labels, attributes = train_loader.get_samples()
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        attributes = attributes.to(args.device)
        if args.algorithm == "mixup":
            from training.util import mixup_data

            inputs, yi, yj, lam = mixup_data(inputs, labels, args.device)

        # Forward pass
        outputs = model(inputs)

        # Compute metrics and update tracker
        if args.algorithm == "mixup":
            loss = lam * criterion(outputs, yi) + (1 - lam) * criterion(outputs, yj)
        else:
            loss = criterion(outputs, labels)

        metric_tracker.update("train", task_id, task_id, loss, outputs, labels, attributes)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if jt_scheduler is not None:
            jt_scheduler.step(loss)
        lr = optimizer.param_groups[0]["lr"]

        iteration_counter += 1
        if args.algorithm == "SWAD":
            client_algorithm.swad_step += 1
            client_algorithm.swad_model.update_parameters(model, step=client_algorithm.swad_step)

        if (iteration_counter % args.training_log_frequency) == 0:
            log_training_results(
                metric_tracker, task_id, iteration_counter, lr, wandb, args.output_dir
            )

    model = model.to("cpu")  # Free up GPU memory
    return model, optimizer, client_dataset, metric_tracker, iteration_counter


def evaluate_on_holdout(
    model, client_dataset, client_algorithm, metrics_tracker, train_task_id, iteration, wandb, args
):
    """
    Evaluates the model on the holdout set.

    Args:
        model (torch.nn.Module): The model.
        client_dataset (object): Client dataset.
        client_algorithm (object): Client algorithm.
        metrics_tracker (ClientTrackers): An object that tracks metrics for a specific client.
        train_task_id (int): The training task ID.
        iteration (int): The current iteration.
        wandb: WandB object for logging.
        args (argparse.Namespace): Parsed command-line arguments.
    """

    task_id = 0  # holdout tracker only keep tracks of a single task (last task)

    model = model.to(args.device)
    criterion = initialize_criterion(args.criterion)
    device = args.device
    model.eval()

    client_dataset_copy = copy.deepcopy(client_dataset)
    holdout_loader = DataLoader(
        client_dataset_copy.holdout_set,
        batch_size=args.eval_batch_size,
        num_workers=args.eval_num_workers,
        shuffle=False,
    )

    # SWAD Test
    if args.algorithm == "SWAD":
        # swad_model = client_algorithm.swad.get_final_model()
        # n_steps = 500 #if not args.debug else 10
        # print(f"Update SWAD BN statistics for {n_steps} steps ...")
        # update_bn(train_loader, swad_model, n_steps, device)

        # The "swad_last_test_model" should have been computed in the evaluate_on_all_tasks() call
        swad_model = client_algorithm.swad_last_test_model.eval()
        model = swad_model.to(device)

    evaluate_split(
        model,
        criterion,
        holdout_loader,
        client_algorithm,
        metrics_tracker,
        device,
        task_id,
        task_id,
        args,
        split="holdout",
    )
    log_holdout_results(metrics_tracker, iteration, wandb, args.output_dir)
    model = model.to("cpu")  # Free up GPU memory


def evaluate_on_seen_tasks(
    model, client_dataset, client_algorithm, metrics_tracker, train_task_id, iteration, args
):
    """
    Evaluates the model on the validation and test sets for all tasks.

    Args:
        model (torch.nn.Module): The model.
        client_dataset (object): Client dataset.
        client_algorithm (object): Client algorithm.
        metrics_tracker (ClientTrackers): An object that tracks metrics for a specific client.
        train_task_id (int): The training task ID.
        iteration (int): The current iteration.
        wandb: WandB object for logging.
        args (argparse.Namespace): Parsed command-line arguments.
    """

    model = model.to(args.device)
    criterion = initialize_criterion(args.criterion)
    device = args.device
    model.eval()
    test_model = model

    client_dataset_copy = copy.deepcopy(client_dataset)

    for eval_task_id in range(train_task_id + 1):
        client_dataset_copy.load_task(eval_task_id)

        val_loader = DataLoader(
            client_dataset_copy.val_set,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            shuffle=False,
        )

        evaluate_split(
            model,
            criterion,
            val_loader,
            client_algorithm,
            metrics_tracker,
            device,
            train_task_id,
            eval_task_id,
            args,
            split="val",
            enable_SWAD=False,
        )
    return metrics_tracker.average_loss_over_seen_tasks("val", train_task_id)


def evaluate_on_all_tasks(
    model,
    client_dataset,
    train_loader,
    client_algorithm,
    metrics_tracker,
    train_task_id,
    iteration,
    is_predict,
    wandb,
    args,
):
    """
    Evaluates the model on the validation and test sets for all tasks.

    Args:
        model (torch.nn.Module): The model.
        client_dataset (object): Client dataset.
        train_loader (torch.utils.data.DataLoader): The training data loader for the dataset.
        client_algorithm (object): Client algorithm.
        metrics_tracker (ClientTrackers): An object that tracks metrics for a specific client.
        train_task_id (int): The training task ID.
        iteration (int): The current iteration.
        is_predict (bool): whether to predict on test set
        wandb: WandB object for logging.
        args (argparse.Namespace): Parsed command-line arguments.
    """

    model = model.to(args.device)
    criterion = initialize_criterion(args.criterion)
    device = args.device
    model.eval()
    test_model = model

    client_dataset_copy = copy.deepcopy(client_dataset)

    for eval_task_id in range(args.num_tasks):
        client_dataset_copy.load_task(eval_task_id)

        val_loader = DataLoader(
            client_dataset_copy.val_set,
            batch_size=args.eval_batch_size,
            num_workers=args.eval_num_workers,
            shuffle=False,
        )

        evaluate_split(
            model,
            criterion,
            val_loader,
            client_algorithm,
            metrics_tracker,
            device,
            train_task_id,
            eval_task_id,
            args,
            split="val",
        )
        log_evaluation_results(
            "val", metrics_tracker, train_task_id, eval_task_id, iteration, wandb, args.output_dir
        )

    if is_predict:
        for eval_task_id in range(args.num_tasks):
            client_dataset_copy.load_task(eval_task_id)

            test_loader = DataLoader(
                client_dataset_copy.test_set,
                batch_size=args.eval_batch_size,
                num_workers=args.eval_num_workers,
                shuffle=False,
            )
            # SWAD Test
            if (
                args.algorithm == "SWAD" and eval_task_id == 0
            ):  # Run only on the first iteration of the loop
                swad_model = client_algorithm.swad.get_final_model()
                n_steps = 500  # if not args.debug else 10
                print(f"Update SWAD BN statistics for {n_steps} steps ...")
                update_bn(train_loader, swad_model, n_steps, device)
                test_model = swad_model.to(device).eval()
                client_algorithm.swad_last_test_model = test_model
            evaluate_split(
                test_model,
                criterion,
                test_loader,
                client_algorithm,
                metrics_tracker,
                device,
                train_task_id,
                eval_task_id,
                args,
                split="test",
            )
            log_evaluation_results(
                "test",
                metrics_tracker,
                train_task_id,
                eval_task_id,
                iteration,
                wandb,
                args.output_dir,
            )

    model = model.to("cpu")  # Free up GPU memory
    test_model = test_model.to("cpu")  # Free up GPU memory


def evaluate_split(
    model,
    criterion,
    eval_loader,
    client_algorithm,
    metrics_tracker,
    device,
    train_task_id,
    eval_task_id,
    args,
    split,
):
    """
    Evaluates the model on the specified data loader.

    Args:
        model (torch.nn.Module): The model.
        criterion: The loss function criterion.
        eval_loader (torch.utils.data.DataLoader): The data loader for the dataset.
        client_algorithm (object): Client algorithm.
        metrics_tracker (ClientTrackers): An object that tracks metrics for a specific client.
        device: device to use for evaluation.
        train_task_id (int): The training task ID.
        eval_task_id (int): The evaluation task ID.
        args (argparse.Namespace): Parsed command-line arguments.
        split (str): The split to compute the loss from (train, val, test or holdout).
    """
    metrics_tracker.trackers[split].reset_task_metrics(train_task_id, eval_task_id)

    with torch.no_grad():
        for inputs, labels, attributes in eval_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            attributes = attributes.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute metrics and update tracker
            loss = criterion(outputs, labels)
            metrics_tracker.update(
                split, train_task_id, eval_task_id, loss, outputs, labels, attributes
            )

    # SWAD Valdiation (on the same training task)
    if args.algorithm == "SWAD" and split == "val" and train_task_id == eval_task_id:
        task_val_acc, task_val_macro_acc, _ = metrics_tracker.average_acc(
            split, train_task_id, eval_task_id
        )
        task_val_loss = metrics_tracker.average_loss(split, train_task_id, eval_task_id)

        client_algorithm.swad.update_and_evaluate(
            client_algorithm.swad_model, task_val_macro_acc, task_val_loss
        )

        if hasattr(client_algorithm.swad, "dead_valley") and client_algorithm.swad.dead_valley:
            print("SWAD valley is dead -> early stop !")
            client_algorithm.swad_early_stop = True

        client_algorithm.swad_model = AveragedModel(model)  # reset
