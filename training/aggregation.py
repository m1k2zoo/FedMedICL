from helpers.client_management import save_models_and_optimizers
import random
import torch
import copy


def average_state_dicts(state_dicts, weights=None):
    """
    Averages the state_dicts, optionally weighted by the provided weights.

    Args:
        state_dicts (list): List of state_dicts to average.
        weights (list, optional): List of weights corresponding to each state_dict's importance. Defaults to None.

    Returns:
        dict: Averaged (and possibly weighted) state_dict.
    """
    keys = state_dicts[0].keys()
    if weights is None:
        weights = [1] * len(state_dicts)  # Equal weighting if none provided

    else:
        weights = weights.tolist()

    # Ensure the weights sum to 1 for proper averaging
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]

    # Unweighted Averaging (Old Version)
    # values = zip(*map(lambda dict: dict.values(), state_dicts))
    # averaged_state_dict = {key: sum(value)/len(state_dicts) for key, value in zip(keys, values)}

    values = zip(
        *[
            map(lambda state: state[1] * normalized_weights[i], enumerate(dict.values()))
            for i, dict in enumerate(state_dicts)
        ]
    )
    averaged_state_dict = {key: sum(value) for key, value in zip(keys, values)}

    return averaged_state_dict


def backbone_averaging(models, weights=None):
    """
    Performs averaging specifically on the backbone layers of the provided models.

    Args:
        models (list): List of models to average.
        weights (list, optional): Weights for each model's contribution. Defaults to None.

    Returns:
        dict: Averaged (and possibly weighted) state_dict of the backbone layers.
    """
    backbone_state_dicts = [model.backbone_layers.state_dict() for model in models]
    averaged_backbone_state_dict = average_state_dicts(backbone_state_dicts, weights)

    return averaged_backbone_state_dict


def complete_model_averaging(models, weights=None):
    """
    Performs averaging on all layers of the provided models.

    Args:
        models (list): List of models to average.
        weights (list, optional): Weights for each model's contribution. Defaults to None.

    Returns:
        tuple: A tuple containing two dictionaries, each representing the
               averaged state_dicts for the backbone_layers and output_layer respectively.
    """
    # Extract state_dicts for each type of layer
    backbone_layer_dicts = [model.backbone_layers.state_dict() for model in models]
    output_layer_dicts = [model.output_layer.state_dict() for model in models]

    # Average each set of state_dicts
    avg_backbone_layers = average_state_dicts(backbone_layer_dicts, weights)
    avg_output_layer = average_state_dicts(output_layer_dicts, weights)

    return avg_backbone_layers, avg_output_layer


def federated_aggregation(args, models, optimizers, client_subdirs=None, weights=None):
    """
    Perform federated aggregation by averaging model weights based on the specified approach.

    Args:
        args (argparse.Namespace): Command-line arguments or configuration settings.
        models (List[Model]): List of models to aggregate.
        optimizers (List[Optimizer]): List of optimizers corresponding to the models.
        client_subdirs (List[str]): List of subdirectories for saving the updated models.
        weights (list, optional): Weights for each model's contribution. Defaults to None.

    Returns:
        List[Model]: Updated list of models after aggregation.
    """
    # Determine the averaging approach based on the flag in args
    if args.average_all_layers:
        avg_backbone_layers, avg_output_layer = complete_model_averaging(models, weights)
        # Update all layers in each model
        models = [
            model.replace_all_layers(avg_backbone_layers, avg_output_layer) for model in models
        ]

    else:
        # Averaging only the backbone layers
        averaged_backbone_dict = backbone_averaging(models, weights)
        # Update the backbone layer of each model with the averaged backbone
        models = [model.replace_backbone_layers(averaged_backbone_dict) for model in models]

    if args.use_distributed:
        # Save the updated models
        save_models_and_optimizers(models, optimizers, client_subdirs)

    return models


def daisy_chaining(models, optimizers):
    """
    Shuffles the models and optimizers lists in place, maintaining the correspondence between them.

    Args:
        models (List[Model]): A list of models.
        optimizers (List[Optimizer]): A list of optimizers corresponding to the models.

    Returns:
        A tuple of the shuffled models and optimizers lists.
    """
    # Ensure the lists are the same length
    assert len(models) == len(optimizers), "models and optimizers must have the same length"

    # Generate a random permutation of indices
    indices = list(range(len(models)))
    random.shuffle(indices)

    # Reorder both lists according to the permutation
    shuffled_models = [models[i] for i in indices]
    shuffled_optimizers = [optimizers[i] for i in indices]

    return shuffled_models, shuffled_optimizers


def test_federated_aggregation(args, initial_models, optimizers, client_subdirs=None):
    """
    Test the federated_aggregation function under two scenarios:
    1. When args.average_all_layers is True
    2. When args.average_all_layers is False

    This function checks if the aggregation is performed correctly.

    Args:
        args (argparse.Namespace): Command-line arguments or configuration settings.
        initial_models (List[FoundationModel]): List of models to aggregate.
        optimizers (List[Optimizer]): List of optimizers corresponding to the models.
        client_subdirs (List[str]): Subdirectories for saving models (if applicable).
    """

    # First, ensure that the models initially have different weights
    for i in range(len(initial_models) - 1):
        for param1, param2 in zip(
            initial_models[i].parameters(), initial_models[i + 1].parameters()
        ):
            if torch.equal(param1.data, param2.data):
                print("Failed! Initial models have the same weights.")
                return

    # Case 1: Test with args.average_all_layers = True
    print("Testing with args.average_all_layers = True...")
    args.average_all_layers = True
    models = copy.deepcopy(
        initial_models
    )  # Create a deep copy to avoid modifying the original models
    aggregated_models = federated_aggregation(args, models, optimizers, client_subdirs)

    # Check if the weights are the same after aggregation
    model1_weights = aggregated_models[0].state_dict()
    for i in range(1, len(aggregated_models)):
        model2_weights = aggregated_models[i].state_dict()
        for weight1, weight2 in zip(model1_weights.items(), model2_weights.items()):
            if not torch.equal(weight1[1], weight2[1]):
                print(
                    f"Failed! Aggregation failed with average_all_layers=True. Weight: {weight1[0]}"
                )
                return
    print("Success for case with average_all_layers=True")

    # Case 2: Test with args.average_all_layers = False
    print("Testing with args.average_all_layers = False...")
    args.average_all_layers = False
    models = copy.deepcopy(initial_models)  # Reset with original models
    aggregated_models = federated_aggregation(args, models, optimizers, client_subdirs)

    # Check if only the backbone layers' weights are the same after aggregation
    model1_weights = aggregated_models[0].backbone_layers.state_dict()
    for i in range(1, len(aggregated_models)):
        model2_weights = aggregated_models[i].backbone_layers.state_dict()
        for weight1, weight2 in zip(model1_weights.items(), model2_weights.items()):
            if not torch.equal(weight1[1], weight2[1]):
                print(
                    f"Failed! Aggregation failed with average_all_layers=False. Weight: {weight1[0]}"
                )
                return

    # Confirm that other layers (e.g., output_layer) have not been averaged/aggregated
    model1_output_weights = aggregated_models[0].output_layer.state_dict()
    for i in range(1, len(aggregated_models)):
        model2_output_weights = aggregated_models[i].output_layer.state_dict()
        for weight1, weight2 in zip(model1_output_weights.items(), model2_output_weights.items()):
            if torch.equal(weight1[1], weight2[1]):
                print(
                    f"Failed! Unexpected aggregation in output_layer with average_all_layers=False. Weight: {weight1[0]}"
                )
                return

    print("Success for case with average_all_layers=False")


def compute_fairfed_weights(
    acc_global_t, acc_k_t, weights_prev_t, beta, F_global_t=None, F_k_t=None
):
    """
    Computes the updated weights for each client based on the FairFed paper.

    Parameters:
    - acc_global_t (float): The global accuracy of the model at time t.
    - acc_k_t (Tensor): A tensor of accuracies for each client's model at time t.
    - weights_prev_t (Tensor): A tensor of the previous weights for each client at time t-1.
    - beta (float): A hyperparameter that controls the learning rate of the weight updates.
    - F_global_t (float, optional): The fairness metric of the global model at time t, required if F_k_t is provided.
    - F_k_t (Tensor, optional): A tensor of fairness metrics for each client's model at time t.

    Returns:
    - weights_t (Tensor): A tensor representing the normalized updated weights for each client at time t.

    Note:
    - All input tensors must be of the same length.
    - F_global_t and F_k_t should either both be None or both be provided.
    """
    K = len(weights_prev_t)  # Number of clients

    # Compute Δᵗₖ based on whether Fᵗₖ is defined
    if F_k_t is not None:
        delta_k = torch.abs(F_global_t - F_k_t)
    else:
        delta_k = torch.abs(acc_k_t - acc_global_t)

    # Compute the weight updates
    weight_updates = -beta * (delta_k - torch.mean(delta_k))

    # Update the weights
    weights_tilde_t = weights_prev_t + weight_updates

    # Normalize the weights to sum to 1
    weights_t = weights_tilde_t / torch.sum(weights_tilde_t)

    return weights_t


def compute_weighted_average_accuracy(acc_k_t, client_dataset_list):
    """
    Computes the weighted average accuracy for a federated learning setup.

    Each client's accuracy is weighted by the size of its dataset.

    Parameters:
    - acc_k_t (list of float): A list containing the accuracy of each client.
    - client_dataset_list (list): A list containing the dataset for each client.

    Returns:
    - float: The weighted average accuracy across all clients.
    """
    # Ensure the input lists are of the same length
    assert len(acc_k_t) == len(client_dataset_list), "Lists must be of the same length."

    # Initialize variables to store the weighted sum of accuracies and total dataset size
    weighted_acc_sum = 0
    total_dataset_size = 0

    # Iterate through each client to calculate the weighted accuracy
    for client_id, accuracy in enumerate(acc_k_t):
        dataset_size = len(client_dataset_list[client_id].train_set)
        weighted_acc_sum += accuracy * dataset_size
        total_dataset_size += dataset_size

    # Compute the global accuracy as the weighted sum of accuracies divided by the total dataset size
    acc_global_t = weighted_acc_sum / total_dataset_size if total_dataset_size > 0 else 0

    return acc_global_t
