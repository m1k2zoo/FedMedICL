import torch.nn as nn

import torch

from collections import Counter
import torch.nn.functional as F
import numpy as np


def applyNormalizer(model, normalizer):
    """
    Apply a normalizer to a neural network model's classifier layer's weights and check if the weights are normalized.

    Args:
        model (torch.nn.Module): The neural network model.
        normalizer (Normalizer): An instance of a Normalizer class with an `apply_on` method.

    Returns:
        None

    This function applies the specified normalizer to the classifier layer's weights of a neural network model. It first
    records the original weights, applies the normalizer, and then checks if the weights have been successfully
    normalized. Optionally, it can also print the L2 norms of weight vectors before and after normalization for
    verification.
    """

    # Access the classifier layer's weights before normalization
    original_weights = model.output_layer.weight.data.clone()

    # Apply the Normalizer
    normalizer.apply_on(model)

    # Access the classifier layer's weights after normalization
    normalized_weights = model.output_layer.weight.data

    # Check if the weights are normalized
    is_normalized = check_normalization(normalized_weights)

    if is_normalized:
        print("Weights successfully normalized.")
    else:
        print("Normalization failed.")

    # Optionally, you can print the norms before and after for verification
    original_norms = torch.linalg.norm(
        original_weights.reshape((original_weights.shape[0], -1)), ord=2, dim=1
    )
    normalized_norms = torch.linalg.norm(
        normalized_weights.reshape((normalized_weights.shape[0], -1)), ord=2, dim=1
    )
    print(f"Original norms of weight vectors: {original_norms}")
    print(f"Norms of weight vectors after normalization: {normalized_norms}")


def calculate_cb_loss(
    labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma, device
):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()
    labels_one_hot = labels_one_hot.to(device)

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1).to(device)
    weights = weights * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    pred = logits.softmax(dim=1)
    cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss


class CBLoss(nn.Module):
    def __init__(
        self,
        samples_per_cls,
        num_classes,
        loss_type="softmax",
        beta=0.9999,
        gamma=2.0,
        device=torch.device("cpu"),
    ):
        """
        Initialize the CBLoss class.

        Args:
            samples_per_cls (list): A list containing the number of samples per class.
            num_classes (int): The number of classes.
            loss_type (str): The type of loss to be used ("softmax", "sigmoid", "focal").
            beta (float): Hyperparameter for class balanced loss.
            gamma (float): Hyperparameter for focal loss.
            device (torch.device): The device to run the loss calculation on.
        """
        super(CBLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.device = device

    def forward(self, logits, labels):
        """
        Compute and return the class balanced loss.

        Args:
            logits (torch.Tensor): The raw, unnormalized scores for each class.
            labels (torch.Tensor): The ground truth class labels.

        Returns:
            torch.Tensor: The computed class balanced loss.
        """
        return calculate_cb_loss(
            labels,
            logits,
            self.samples_per_cls,
            self.num_classes,
            self.loss_type,
            self.beta,
            self.gamma,
            self.device,
        )


def count_samples_per_class(labels, num_classes):
    """
    Count the number of samples per class in a dataset provided by a dataloader.

    Parameters:
    labels (list): List of labels in the dataset.
    num_classes (int): The total number of classes in the dataset.

    Returns:
    list: A list of length num_classes with the count of samples in each class.
    """
    # Initialize a Counter object to count class occurrences
    class_count = Counter()

    # Count occurrences of each class
    class_count.update(labels)

    # Initialize a list with zeros for each class
    samples_per_cls = [0] * num_classes

    # Update the list with the actual counts
    for class_idx, count in class_count.items():
        samples_per_cls[class_idx] = count

    return samples_per_cls


def check_normalization(weights, LpNorm=2, tolerance=1e-5):
    """
    Function to check if the weights of a layer are normalized.

    Parameters:
    weights (torch.Tensor): Weight tensor of a layer.
    LpNorm (int): The p-value for the Lp-norm used in normalization.
    tolerance (float): Tolerance for the normalization check.

    Returns:
    bool: True if weights are normalized, False otherwise.
    """
    # Reshaping the weight tensor and calculating its Lp norm
    weights_vec = weights.reshape((weights.shape[0], -1))
    norms = torch.linalg.norm(weights_vec, ord=LpNorm, dim=1)

    # Checking if all norms are close to 1 within a specified tolerance
    return torch.all(torch.abs(norms - 1.0) < tolerance)


class Normalizer:
    """
    A class used to normalize the weight vectors of neurons in the classifier layer
    of a neural network model to ensure balanced contributions from each neuron.

    Attributes
    ----------
    LpNorm : int
        the p-value for the Lp-norm used in normalization (default is 2).
    tau : float
        a scalar multiplier applied to the norm (default is 1).

    Methods
    -------
    apply_on(model)
        Applies tau-normalization on the classifier layer's weight vectors of the given model.
    """

    def __init__(self, LpNorm=2, tau=1):
        """
        Initializes the Normalizer with specified Lp-norm and tau value.

        Parameters:
        LpNorm (int): The p-value for the Lp-norm used in normalization.
        tau (float): A scalar multiplier applied to the norm.
        """
        self.LpNorm = LpNorm
        self.tau = tau

    def apply_on(self, model):
        """
        Applies tau-normalization on the classifier layer's weight vectors of the given model.

        Parameters:
        model (torch.nn.Module): The target neural network model.
            Should be an instance of a class that has an 'output_layer' attribute.

        Raises:
        AttributeError: If the provided model does not have an 'output_layer' attribute.
        """
        if not hasattr(model, "output_layer"):
            raise AttributeError("The provided model does not have an 'output_layer' attribute.")

        # Access the weight tensor of the classifier layer in the provided model instance
        curLayer = model.output_layer.weight

        curparam = curLayer.data

        curparam_vec = curparam.reshape((curparam.shape[0], -1))
        neuronNorm_curparam = (
            (torch.linalg.norm(curparam_vec, ord=self.LpNorm, dim=1) ** self.tau)
            .detach()
            .unsqueeze(-1)
        )
        scalingVect = torch.ones_like(curparam)

        idx = neuronNorm_curparam == neuronNorm_curparam
        idx = idx.squeeze()
        tmp = 1 / (neuronNorm_curparam[idx].squeeze())
        for _ in range(len(scalingVect.shape) - 1):
            tmp = tmp.unsqueeze(-1)

        scalingVect[idx] = torch.mul(scalingVect[idx], tmp)
        curparam[idx] = scalingVect[idx] * curparam[idx]
