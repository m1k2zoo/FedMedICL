import os
from tqdm import tqdm

from helpers.class_balancing import applyNormalizer, Normalizer
from helpers.client_management import *
from training.aggregation import (
    federated_aggregation,
    daisy_chaining,
    compute_fairfed_weights,
    compute_weighted_average_accuracy,
)
from training.train import train_client, evaluate_on_all_tasks, evaluate_on_holdout
from training.util import (
    InfiniteDataLoader,
    initialize_classifier_optimizer,
    ClientAlgorithm,
    FairFed,
)
from training.replay_management import ClientBuffer


def train_all_clients(
    models, optimizers, client_dataset_list, client_trackers, args, use_fl, wandb
):
    """
    Train all clients on all tasks, with either distributed training or centralized training

    Args:
        models (list): List of client models.
        optimizers (list): List of client optimizers.
        client_dataset_list (list): List of client datasets.
        client_trackers (list): List of client trackers.
        args (argparse.Namespace): Command-line arguments.
        use_fl (bool): Whether to use federated learning.
        wandb (wandb.sdk.wandb_run.Run): WandB run object.

    Returns:
        client_trackers (list): List of client trackers after training.
    """

    if args.use_distributed:
        # Create a subdirectory for each client's dataset
        client_subdirs = []
        for i, _ in enumerate(client_dataset_list):
            client_subdir = os.path.join(args.output_dir, f"client_{i}")
            os.makedirs(client_subdir, exist_ok=True)
            client_subdirs.append(client_subdir)

        # Save each client's dataset, tracker, model, and optimizer
        save_datasets_and_trackers(client_dataset_list, client_trackers, client_subdirs)
        save_models_and_optimizers(models, optimizers, client_subdirs)

        # Train the clients for all tasks on SLURM and load their metrics
        train_distributed(
            models, optimizers, client_dataset_list, client_subdirs, args, use_fl, wandb
        )
        client_trackers = load_client_trackers(args.output_dir)

    else:
        # Train the clients one by one in a for loop
        client_trackers = train_centralized(
            models, optimizers, client_dataset_list, client_trackers, args, use_fl, wandb
        )

    return client_trackers


def train_centralized(
    models, optimizers, client_dataset_list, client_trackers, args, use_fl, wandb, jt_scheduler=None
):
    """
    Function for training client models in centralized mode.

    Args:
        models (list): List of client models.
        optimizers (list): List of client optimizers.
        client_dataset_list (list): List of client datasets.
        client_subdirs (list): List of client subdirectories.
        num_tasks (int): Number of tasks.
        args (argparse.Namespace): Parsed command-line arguments.
        use_fl (boolean): Whether to use federated learning.
        wandb (wandb.run, optional): wandb run object for logging.
        jt_scheduler (torch.optim.lr_scheduler._LRScheduler, optional): A learning rate scheduler for joint training.

    Returns:
        List[Model]: Updated list of client trackers after the training.
    """
    iteration_counters = [0] * len(client_dataset_list)
    evaluation_frequency = max(int(args.num_rounds / args.num_evaluations), 1)
    last_round = args.num_rounds - 1

    # Create a dictionary to store buffers for each client
    clients_buffers = {
        client_dataset.client_id: ClientBuffer(client_dataset.train_set)
        for client_dataset in client_dataset_list
    }

    if args.algorithm == "fairfed":
        fairfed = FairFed(client_dataset_list, args.device)
        # For FedFair, we need to:
        # keep track of acc_global_t, acc_k_t when doing evaluation
        # use compute_fairfed_weights(acc_global_t, acc_k_t, fairfed.weights_prev_t, fairfed.beta) when federated averaging
        # update fairfed.weights_prev_t

    clients_algorithm = []
    for client_dataset in tqdm(client_dataset_list):
        client_id = client_dataset.client_id
        model = models[client_id]
        client_algorithm = ClientAlgorithm(model, args.device, is_swad=args.algorithm == "SWAD")
        clients_algorithm.append(client_algorithm)

    for task_id in tqdm(range(args.num_tasks), desc=f"Tasks"):
        print(f"Task {task_id + 1}/{args.num_tasks}")

        # Reset SWAD early_stop flag at the begining of each task
        if args.algorithm == "SWAD" and task_id > 0:
            for client_dataset in tqdm(client_dataset_list):
                client_id = client_dataset.client_id
                model = models[client_id]
                clients_algorithm[client_id].reset_SWAD(model)
                # clients_algorithm[client_id].swad_early_stop = False
                # client_algorithm[client_id].swad.dead_valley = False

        # Construct the infinite training loaders for the current task
        client_infinite_loaders = {}
        print("Creating clients training dataloaders...")
        for client_dataset in tqdm(client_dataset_list):
            client_id = client_dataset.client_id
            client_dataset.load_task(task_id)
            if args.algorithm == "resampling":
                dataloader_balancing_type = "group"
            else:
                dataloader_balancing_type = "target"
            infinite_loader = InfiniteDataLoader(
                client_dataset.train_set,
                clients_buffers[client_id],
                args.batch_size,
                args.num_workers,
                use_rehearsal=args.use_rehearsal,
                enable_balancing=args.dataloader_balancing,
                imbalance_type=dataloader_balancing_type,
            )  # args.balanced_rehearsal)
            client_infinite_loaders[client_id] = infinite_loader

        for fl_round in tqdm(range(args.num_rounds), desc=f"FL Rounds", position=0):
            # print("Training clients...")
            for client_dataset in client_dataset_list:
                if args.algorithm == "SWAD" and clients_algorithm[client_id].swad_early_stop:
                    print("SWAD early stop! Skip training")
                else:
                    client_id = client_dataset.client_id
                    metric_tracker = client_trackers[client_id]
                    train_loader = client_infinite_loaders[client_id]

                    # Select the corresponding model for the client
                    model = models[client_id]
                    optimizer = optimizers[client_id]

                    # BEGIN local client training
                    (
                        model,
                        optimizer,
                        client_dataset,
                        metric_tracker,
                        iteration_counters[client_id],
                    ) = train_client(
                        task_id,
                        client_id,
                        iteration_counters[client_id],
                        model,
                        optimizer,
                        client_dataset,
                        clients_algorithm[client_id],
                        metric_tracker,
                        train_loader,
                        args,
                        wandb,
                        jt_scheduler,
                    )

                    # Update the model, tracker, and optimizer lists
                    models[client_id] = model
                    optimizers[client_id] = optimizer

                    if args.algorithm == "CRT":
                        metric_tracker = client_trackers[client_id]  # Don't need to track stage 1
                        infinite_loader = InfiniteDataLoader(
                            client_dataset.train_set,
                            clients_buffers[client_id],
                            args.batch_size,
                            args.num_workers,
                            use_rehearsal=args.use_rehearsal,
                            enable_balancing=args.dataloader_balancing,
                            imbalance_type=dataloader_balancing_type,
                        )  # args.balanced_rehearsal)
                        # second stage training
                        stage2_optimizer = initialize_classifier_optimizer(model, optimizer)
                        (
                            model,
                            stage2_optimizer,
                            client_dataset,
                            metric_tracker,
                            iteration_counters[client_id],
                        ) = train_client(
                            task_id,
                            client_id,
                            iteration_counters[client_id],
                            model,
                            stage2_optimizer,
                            client_dataset,
                            clients_algorithm[client_id],
                            metric_tracker,
                            train_loader,
                            args,
                            wandb,
                            jt_scheduler,
                        )

                        # Unfreeze all parameters in the backbone layers
                        for param in model.backbone_layers.parameters():
                            param.requires_grad = True

                        # Update the model, tracker, but not the optimizer (will use stage 1 optimizer later)
                        models[client_id] = model

                    client_trackers[client_id] = metric_tracker

            # End of local training for all clients
            if use_fl:
                if args.algorithm == "fairfed":
                    print("Evaluating clients as part of fairfed...\n")
                    is_predict = False
                    for client_dataset in tqdm(client_dataset_list):
                        client_id = client_dataset.client_id
                        train_loader = client_infinite_loaders[client_id]

                        evaluate_on_all_tasks(
                            models[client_id],
                            client_dataset,
                            train_loader,
                            clients_algorithm[client_id],
                            client_trackers[client_id],
                            task_id,
                            iteration_counters[client_id],
                            is_predict,
                            wandb,
                            args,
                        )
                        fairfed.acc_k_t[client_id] = client_trackers[client_id].average_acc(
                            "val", task_id, task_id
                        )[0]

                    fairfed.acc_global_t = compute_weighted_average_accuracy(
                        fairfed.acc_k_t, client_dataset_list
                    )
                    fairfed.weights_prev_t = compute_fairfed_weights(
                        fairfed.acc_global_t, fairfed.acc_k_t, fairfed.weights_prev_t, fairfed.beta
                    )

                if args.apply_normalizer:
                    print("Applying Normalizer before federated averaging")
                    # Apply the Normalizer with tau=1 (L2 normalization)
                    normalizer = Normalizer(LpNorm=2, tau=1)
                    for model in models:
                        applyNormalizer(model, normalizer)

                if args.algorithm == "feddc":
                    if fl_round % args.feddc_daisy == args.feddc_daisy - 1:  # daisy chaining
                        # print("Performing daisy chaining")
                        models, optimizers = daisy_chaining(models, optimizers)
                    if fl_round % args.feddc_aggregate == args.feddc_aggregate - 1:  # aggregation
                        # print("Performing federated_aggregation")
                        models = federated_aggregation(
                            args, models, optimizers, client_subdirs=None
                        )

                elif args.algorithm == "fairfed":
                    # print("Performing FedFair aggregation")
                    models = federated_aggregation(
                        args,
                        models,
                        optimizers,
                        client_subdirs=None,
                        weights=fairfed.weights_prev_t,
                    )

                else:
                    # print("Performing federated_aggregation")
                    models = federated_aggregation(args, models, optimizers, client_subdirs=None)

            # Evaluate each client's model
            if fl_round == last_round or (fl_round + 1) % evaluation_frequency == 0:
                print("Evaluating clients...\n")
                is_predict = fl_round == last_round
                for client_dataset in tqdm(client_dataset_list):
                    client_id = client_dataset.client_id
                    train_loader = client_infinite_loaders[client_id]

                    evaluate_on_all_tasks(
                        models[client_id],
                        client_dataset,
                        train_loader,
                        clients_algorithm[client_id],
                        client_trackers[client_id],
                        task_id,
                        iteration_counters[client_id],
                        is_predict,
                        wandb,
                        args,
                    )

            # Special case for novel disease experiment, evaluate on holdout at the end of every task (except last one which will be done later)
            if args.is_novel_disease and task_id < (args.num_tasks - 1) and fl_round == last_round:
                print("Evaluating clients for novel diease (holdout)\n")
                for client_dataset in client_dataset_list:
                    client_id = client_dataset.client_id
                    train_loader = client_infinite_loaders[client_id]

                    evaluate_on_holdout(
                        models[client_id],
                        client_dataset,
                        clients_algorithm[client_id],
                        client_trackers[client_id],
                        task_id,
                        iteration_counters[client_id],
                        wandb,
                        args,
                    )

        if args.use_rehearsal:
            # At the end of the task, update the client's buffer
            for client_dataset in client_dataset_list:
                client_id = client_dataset.client_id
                clients_buffers[client_id].add_task_data(client_dataset.train_set.dataframe)

    print("Evaluating clients (holdout)\n")
    for client_dataset in client_dataset_list:
        client_id = client_dataset.client_id
        evaluate_on_holdout(
            models[client_id],
            client_dataset,
            clients_algorithm[client_id],
            client_trackers[client_id],
            task_id,
            iteration_counters[client_id],
            wandb,
            args,
        )

    return client_trackers


def train_distributed(models, optimizers, client_dataset_list, client_subdirs, args, use_fl, wandb):
    """
    Function for training client models in distributed mode.

    Args:
        models (list): List of client models.
        optimizers (list): List of client optimizers.
        client_dataset_list (list): List of client datasets.
        client_subdirs (list): List of client subdirectories.
        args (argparse.Namespace): Parsed command-line arguments.
        use_fl (boolean): Whether to use federated learning.
        wandb (wandb.run, optional): wandb run object for logging. Defaults to None.
    """
    # Train each client model on its corresponding dataset for each task
    for task_id in tqdm(range(args.num_tasks), desc=f"Tasks"):
        if args.use_wandb:
            wandb.log(f"Task {task_id + 1}/{args.num_tasks}")
        else:
            print(f"Task {task_id + 1}/{args.num_tasks}")

        for fl_round in tqdm(range(args.num_rounds), desc=f"FL Rounds"):
            job_ids = []
            for client_dataset in client_dataset_list:
                client_id = client_dataset.client_id

                # Begin local client training
                process = run_client_job(task_id, client_id, args.output_dir)
                stdout, stderr = process.communicate()

                # Get job ID from sbatch output
                job_id = int(stdout.decode().split()[-1])
                job_ids.append((job_id, client_id))

            # Wait for all client processes to finish
            for job_id, client_id in job_ids:
                print(f"Waiting for task {task_id} and client {client_id}")
                wait_for_job_completion(job_id, client_id, task_id)

            # End of local training for all clients
            save_dataloader_state(fl_round, args)

            # Load the models the newly trained locally and perform federated aggregation
            models = load_models(args.output_dir)

            if use_fl:
                models = federated_aggregation(args, models, optimizers, client_subdirs)
