import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_accuracy(correct, total):
    """
    Computes the accuracy percentage given the number of correctly predicted instances and the total instances.
    """
    if total > 0:
        return round(100.0 * correct / total, 3)
    return 0.0


def compute_per_group_accuracies(group_correct, group_total, num_groups):
    """
    Computes per-group accuracy percentages based on arrays of correct predictions and total instances for each group.
    """
    return {
        group: compute_accuracy(group_correct[group], group_total[group])
        for group in range(num_groups)
    }


def compute_per_category_accuracies(category_correct, category_total, num_categories):
    """
    Computes per-category accuracy percentages based on arrays of correct predictions and total instances for each category.
    """
    return {
        category: compute_accuracy(
            np.sum(category_correct[category]), np.sum(category_total[category])
        )
        for category in range(num_categories)
    }


def compute_macro_accuracy(per_attribute_accuracies):
    """
    Calculate macro accuracy from a dictionary of per-attribute accuracies.

    Parameters:
    per_attribute_accuracies (dict): Dictionary of attribute accuracies.

    Returns:
    float: Macro accuracy.
    """
    return round(sum(per_attribute_accuracies.values()) / len(per_attribute_accuracies), 3)


def compute_auc(true_labels, predicted_scores, num_categories):
    """
    Computes the AUC (Area Under the ROC Curve) using true labels and predicted scores.

    Args:
        true_labels (numpy array): True labels of the dataset.
        predicted_scores (numpy array): Predicted scores or probabilities for each class.
        num_categories (int): Number of categories/classes in the dataset.

    Returns:
        float: The AUC value.
    """
    if num_categories == 2:
        # Binary classification: assuming the positive class probabilities are in the second column
        return round(roc_auc_score(true_labels, predicted_scores[:, 1]), 3)
    elif num_categories > 2:
        # Multiclass classification: assuming a one-vs-rest strategy
        try:
            return round(
                roc_auc_score(
                    true_labels,
                    predicted_scores,
                    multi_class="ovo",
                    average="macro",
                    labels=list(range(num_categories)),
                ),
                3,
            )
        except ValueError as e:
            print("true_labels:", true_labels)
            print("predicted_scores:", predicted_scores)
            # Handle cases where AUC cannot be computed
            # print(f"Error computing AUC: {e}")
            return None


def compute_per_category_auc(true_labels, predicted_scores, num_categories):
    """
    Computes the AUC (Area Under the ROC Curve) for each class using true labels and predicted scores.

    Args:
        true_labels (numpy array): True labels of the dataset.
        predicted_scores (numpy array): Predicted scores or probabilities for each class.
        num_categories (int): Number of categories/classes in the dataset.

    Returns:
        dict: A dictionary with class indices as keys and their respective AUCs as values.
    """
    if num_categories == 2:
        # For binary classification, return AUC for both the negative and positive class
        auc_score_positive = roc_auc_score(true_labels, predicted_scores[:, 1])
        auc_score_negative = roc_auc_score(true_labels, predicted_scores[:, 0])
        return {0: auc_score_negative, 1: auc_score_positive}
    else:
        # For multiclass classification, compute per-class AUC using a one-vs-rest approach
        auc_scores = {}
        true_labels_binarized = label_binarize(true_labels, classes=range(num_categories))
        for class_index in range(num_categories):
            if (
                true_labels_binarized[:, class_index].sum() > 0
            ):  # Ensure there are positive samples for the class
                try:
                    auc_scores[class_index] = roc_auc_score(
                        true_labels_binarized[:, class_index], predicted_scores[:, class_index]
                    )
                except:
                    auc_scores[class_index] = float("nan")  # Not enough samples to compute AUC
            else:
                auc_scores[class_index] = float("nan")  # Not enough samples to compute AUC
        return auc_scores


class ClientTrackers:
    def __init__(
        self, num_tasks, num_groups, client_id, num_categories, algorithm, use_macro_avg=False
    ):
        """
        Class to track metrics for a specific client in a federated learning paradigm.

        Args:
            num_tasks (int): The number of continual learning tasks.
            num_groups (int): The number of groups covering the sensitive attribute.
            client_id (int): The client ID.
            num_categories (int): The number of categories (or classes) in the dataset.
            algorithm (str): The type of the model.
            use_macro_avg (bool): Flag to indicate if macro-averaged accuracy should be computed. Defaults to False.
        """
        self.trackers = {}
        self.trackers["train"] = MetricsTracker(
            num_tasks, num_groups, num_categories, use_macro_avg, tracker_type="train"
        )
        self.trackers["val"] = MetricsTracker(
            num_tasks, num_groups, num_categories, use_macro_avg, tracker_type="val"
        )
        self.trackers["test"] = MetricsTracker(
            num_tasks, num_groups, num_categories, use_macro_avg, tracker_type="test"
        )
        self.trackers["holdout"] = MetricsTracker(
            1, num_groups, num_categories, use_macro_avg, tracker_type="holdout"
        )
        self.num_tasks = num_tasks
        self.client_id = client_id
        self.use_macro_avg = use_macro_avg
        self.algorithm = algorithm

    def average_loss(self, split, train_task_id, eval_task_id=None):
        """
        Returns the average loss for a given training and evaluation task in the specified split.

        Args:
            split (str): The split to compute the loss from (train, val, or test).
            train_task_id (int): The index of the training task.
            eval_task_id (int, optional): The index of the evaluation task. Required if tracker_type is 'val' or 'test'.

        Returns:
            float: The average loss for the specified task and split.
        """
        if split not in ["train", "val", "test", "holdout"]:
            raise ValueError("Invalid split. Expected 'train', 'val', 'test' or 'holdout'.")

        return self.trackers[split].average_loss(train_task_id, eval_task_id)

    def average_loss_over_seen_tasks(self, split, train_task_id):
        """
        Returns the average loss over a series of evaluation tasks, from 0 up to the given training task id.

        Args:
            split (str): The split to compute the loss from (train, val, or test).
            train_task_id (int): The index of the training task.

        Returns:
            float: The average contioual loss for the specified training task and split.
        """
        if split not in ["train", "val", "test", "holdout"]:
            raise ValueError("Invalid split. Expected 'train', 'val', 'test' or 'holdout'.")
        return self.trackers[split].average_loss_over_seen_tasks(train_task_id)

    def average_acc(self, split, train_task_id, eval_task_id=None):
        """
        Returns the current average accuracy for a given training and evaluation task in the specified split.

        Args:
            split (str): The split to compute the accuracy from (train, val, or test).
            train_task_id (int): The index of the training task.
            eval_task_id (int, optional): The index of the evaluation task. Required if tracker_type is 'val' or 'test'.

            tuple: A tuple containing the average task accuracy and per-group accuracies.
                   The per-group accuracies are represented as a dictionary.
        """
        if split not in ["train", "val", "test", "holdout"]:
            raise ValueError("Invalid split. Expected 'train', 'val', 'test' or 'holdout'.")

        return self.trackers[split].average_acc(train_task_id, eval_task_id)

    def get_auc(self, split, train_task_id, eval_task_id=None):
        """
        Returns the current AUC for a given training and evaluation task in the specified split.

        Args:
            split (str): The split to compute the AUC from (train, val, or test, or holdout).
            train_task_id (int): The index of the training task.
            eval_task_id (int, optional): The index of the evaluation task. Required if tracker_type is 'val' or 'test'.

        Returns:
            float: The average AUC for the specified task and split.
        """
        if split not in ["train", "val", "test", "holdout"]:
            raise ValueError("Invalid split. Expected 'train', 'val', 'test' or 'holdout'.")

        return self.trackers[split].get_auc(train_task_id, eval_task_id)

    def final_acc(self, split):
        """
        Returns the final accuracy for a specific client in the specified split.

        Args:
            split (str): The split to compute the accuracy from (train, val, or test).

        Returns:
            float: The final accuracy for the specified client and split.
        """
        if split not in ["train", "val", "test", "holdout"]:
            raise ValueError("Invalid split. Expected 'train', 'val', 'test' or 'holdout'.")

        return self.trackers[split].final_acc()

    def update(self, split, train_task_id, eval_task_id, loss, outputs, labels, attributes):
        """
        Updates the metrics for a given training and evaluation task in the specified split.

        Args:
            split (str): The split to update the metrics for (train, val, or test).
            train_task_id (int): The index of the current task being trained on.
            eval_task_id (int): The index of the current task being evaluated.
            loss (torch.Tensor): The loss value for the current batch.
            outputs (torch.Tensor): The predicted outputs from the model.
            labels (torch.Tensor): The ground truth labels.
            attributes (torch.Tensor): The attributes indicating the groups.
        """
        if split not in ["train", "val", "test", "holdout"]:
            raise ValueError("Invalid split. Expected 'train', 'val', 'test' or 'holdout'.")

        return self.trackers[split].update(
            train_task_id, eval_task_id, loss, outputs, labels, attributes
        )


class MetricsTracker:
    def __init__(self, num_tasks, num_groups, num_categories, use_macro_avg, tracker_type):
        """
        Class to track classification metrics during continual learning.

        Args:
            num_tasks (int): The number of continual learning tasks.
            num_groups (int): The number of groups covering the sensitive attribute.
            num_categories (int): The number of categories (or classes) in the dataset.
            use_macro_avg (bool): Flag to indicate if macro-averaged accuracy should be computed.
            tracker_type (str): Type of tracker. Expect: "train", "val", "test" or "holdout".
        """
        self.num_groups = num_groups
        self.num_categories = num_categories
        self.num_tasks = num_tasks
        self.use_macro_avg = use_macro_avg
        self.tracker_type = tracker_type

        if self.tracker_type == "train" or self.tracker_type == "holdout":
            self.loss = np.zeros(num_tasks)
            self.correct = np.zeros(num_tasks)
            self.total = np.zeros(num_tasks)
            self.per_group_correct = np.zeros((num_groups, num_tasks))
            self.per_group_total = np.zeros((num_groups, num_tasks))
            self.per_category_correct = np.zeros((num_categories, num_tasks))
            self.per_category_total = np.zeros((num_categories, num_tasks))
            self.predicted_scores = np.empty((0, num_categories))
            self.true_labels = np.array([], dtype=np.int64)
        else:
            self.loss = np.zeros((num_tasks, num_tasks))
            self.correct = np.zeros((num_tasks, num_tasks))
            self.total = np.zeros((num_tasks, num_tasks))
            self.per_group_correct = np.zeros((num_groups, num_tasks, num_tasks))
            self.per_group_total = np.zeros((num_groups, num_tasks, num_tasks))
            self.per_category_correct = np.zeros((num_categories, num_tasks, num_tasks))
            self.per_category_total = np.zeros((num_categories, num_tasks, num_tasks))
            self.predicted_scores = {
                task: np.empty((0, num_categories)) for task in range(num_tasks)
            }
            self.true_labels = {task: np.array([], dtype=np.int64) for task in range(num_tasks)}

    def reset_task_metrics(self, train_task_id, eval_task_id=None):
        """
        Reset the metrics of a particular task.

        Args:
            train_task_id (int): The index of the training task to reset.
            eval_task_id (int, optional): The index of the evaluation task to reset. Only used for val/test tracker types.
        """
        if train_task_id < 0 or train_task_id >= self.num_tasks:
            raise ValueError("Invalid train_task_id")
        if self.tracker_type == "val" or self.tracker_type == "test":
            if eval_task_id is None or eval_task_id < 0 or eval_task_id >= self.num_tasks:
                raise ValueError("Invalid eval_task_id")

        if self.tracker_type == "train" or self.tracker_type == "holdout":
            self.loss[train_task_id] = 0
            self.correct[train_task_id] = 0
            self.total[train_task_id] = 0
            self.per_group_correct[:, train_task_id] = 0
            self.per_group_total[:, train_task_id] = 0
            self.per_category_correct[:, train_task_id] = 0
            self.per_category_total[:, train_task_id] = 0
        else:
            self.loss[train_task_id, eval_task_id] = 0
            self.correct[train_task_id, eval_task_id] = 0
            self.total[train_task_id, eval_task_id] = 0
            self.per_group_correct[:, train_task_id, eval_task_id] = 0
            self.per_group_total[:, train_task_id, eval_task_id] = 0
            self.per_category_correct[:, train_task_id, eval_task_id] = 0
            self.per_category_total[:, train_task_id, eval_task_id] = 0

    def average_loss(self, train_task_id, eval_task_id=None):
        """
        Computes the average loss for a given training and evaluation task. If the tracker type is 'val' or 'test',
        it also requires an evaluation task ID to compute the average loss for that combination.

        Args:
            train_task_id (int): The index of the training task.
            eval_task_id (int, optional): The index of the evaluation task. Required if tracker_type is 'val' or 'test'.

        Returns:
            float: The average loss for the specified task or task combination.
        """
        if self.tracker_type == "train" or self.tracker_type == "holdout":
            ave_loss = np.sum(self.loss[train_task_id]) / np.sum(self.total[train_task_id])
        else:
            ave_loss = np.sum(self.loss[train_task_id, eval_task_id]) / np.sum(
                self.total[train_task_id, eval_task_id]
            )
        return round(ave_loss, 5)

    def average_loss_over_seen_tasks(self, train_task_id):
        """
        Computes the average loss over a series of evaluation tasks, from 0 up to the given training task id.

        Args:
            train_task_id (int): The index of the training task.

        Returns:
            float: The average loss over the seen tasks.
        """
        if self.tracker_type == "train" or self.tracker_type == "holdout":
            # No need for eval_task_id, compute directly
            return self.average_loss(train_task_id)
        else:
            # Iterate over the range of evaluation tasks, compute and accumulate average loss
            total_loss = 0
            for eval_task_id in range(train_task_id + 1):
                total_loss += self.average_loss(train_task_id, eval_task_id)
            # Compute the overall average
            average_loss_over_seen_tasks = total_loss / (train_task_id + 1)
            return round(average_loss_over_seen_tasks, 5)

    def average_acc(self, train_task_id, eval_task_id=None):
        """
        Computes the current average accuracy for a given training and evaluation task. If the tracker type is 'val' or 'test',
        it also requires an evaluation task ID to compute the average accuracy for that combination.

        Args:
            train_task_id (int): The index of the training task.
            eval_task_id (int, optional): The index of the evaluation task. Required if tracker_type is 'val' or 'test'.

        Returns:
            tuple: A tuple containing the average task accuracy, per-category and per-group accuracies.
                The per-group accuracies are represented as a dictionary.
        """
        if self.tracker_type == "train" or self.tracker_type == "holdout":
            accuracy = compute_accuracy(self.correct[train_task_id], self.total[train_task_id])
            per_category_accuracies = compute_per_category_accuracies(
                self.per_category_correct[:, train_task_id],
                self.per_category_total[:, train_task_id],
                self.num_categories,
            )
            per_group_accuracies = compute_per_group_accuracies(
                self.per_group_correct[:, train_task_id],
                self.per_group_total[:, train_task_id],
                self.num_groups,
            )
        else:
            accuracy = compute_accuracy(
                self.correct[train_task_id, eval_task_id], self.total[train_task_id, eval_task_id]
            )
            per_category_accuracies = compute_per_category_accuracies(
                self.per_category_correct[:, train_task_id, eval_task_id],
                self.per_category_total[:, train_task_id, eval_task_id],
                self.num_categories,
            )
            per_group_accuracies = compute_per_group_accuracies(
                self.per_group_correct[:, train_task_id, eval_task_id],
                self.per_group_total[:, train_task_id, eval_task_id],
                self.num_groups,
            )

        return accuracy, per_category_accuracies, per_group_accuracies

    def get_auc(self, train_task_id, eval_task_id=None):
        """
        Computes the current average AUC for a given training and evaluation task. If the tracker type is 'val' or 'test',
        it also requires an evaluation task ID to compute the AUC for that combination.

        Args:
            train_task_id (int): The index of the training task.
            eval_task_id (int, optional): The index of the evaluation task. Required if tracker_type is 'val' or 'test'.

        Returns:
            float: The AUC value.
        """
        try:
            if self.tracker_type == "train" or self.tracker_type == "holdout":
                auc = compute_auc(self.true_labels, self.predicted_scores, self.num_categories)
                per_category_auc = compute_per_category_auc(
                    self.true_labels, self.predicted_scores, self.num_categories
                )
            else:
                auc = compute_auc(
                    self.true_labels[eval_task_id],
                    self.predicted_scores[eval_task_id],
                    self.num_categories,
                )
                per_category_auc = compute_per_category_auc(
                    self.true_labels[eval_task_id],
                    self.predicted_scores[eval_task_id],
                    self.num_categories,
                )

            return auc, per_category_auc
        except AttributeError as e:
            print(f"Error: Missing necessary data attributes in MetricsTracker object - {e}")
            return None, None

    def final_acc(self):
        """
        Computes the final average accuracy for a given client.

        Returns:
            tuple: A tuple containing the final average accuracy, per-category and per-group accuracies.
                The per-group accuracies are represented as a dictionary.
        """
        if self.tracker_type == "train" or self.tracker_type == "holdout":
            total_correct = np.sum(self.correct)
            total_samples = np.sum(self.total)
            per_category_correct_agg = np.sum(self.per_category_correct, axis=1)
            per_category_total_agg = np.sum(self.per_category_total, axis=1)
            per_group_correct_agg = np.sum(self.per_group_correct, axis=1)
            per_group_total_agg = np.sum(self.per_group_total, axis=1)
        else:
            total_correct = np.sum(self.correct)
            total_samples = np.sum(self.total)
            per_category_correct_agg = np.sum(self.per_category_correct, axis=(0, 1))
            per_category_total_agg = np.sum(self.per_category_total, axis=(0, 1))
            per_group_correct_agg = np.sum(self.per_group_correct, axis=(0, 1))
            per_group_total_agg = np.sum(self.per_group_total, axis=(0, 1))

        accuracy = self.compute_accuracy(total_correct, total_samples)
        per_category_accuracies = self.compute_per_category_accuracies(
            per_category_correct_agg, per_category_total_agg, self.num_categories
        )
        per_group_accuracies = self.compute_per_group_accuracies(
            per_group_correct_agg, per_group_total_agg, self.num_groups
        )

        return accuracy, per_category_accuracies, per_group_accuracies

    def update(self, train_task_id, eval_task_id, loss, outputs, labels, attributes):
        """
        Updates the metrics based on the given values for a batch corresponding to a specific task.

        Args:
            train_task_id (int): The index of the current task being trained on.
            eval_task_id (int): The index of the current task being evaluated.
            loss (torch.Tensor): The loss value for the current batch.
            outputs (torch.Tensor): The predicted outputs from the model.
            labels (torch.Tensor): The ground truth labels.
            attributes (torch.Tensor): The attributes indicating the groups.
        """
        probabilities = torch.nn.functional.softmax(outputs.data, dim=1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        # Compute per-group accuracies
        group_correct = np.zeros(self.per_group_correct.shape[0])
        group_total = np.zeros(self.per_group_total.shape[0])
        for group in range(attributes.max().item() + 1):
            mask = attributes == group
            group_correct[group] = (predicted[mask] == labels[mask]).sum().item()
            group_total[group] = mask.sum().item()

        # Update per-category counters if macro-averaging is to be used
        category_correct = np.zeros(self.per_category_correct.shape[0])
        category_total = np.zeros(self.per_category_total.shape[0])
        for category in range(self.per_category_correct.shape[0]):
            mask = labels == category
            category_correct[category] = (predicted[mask] == labels[mask]).sum().item()
            category_total[category] = mask.sum().item()

        if self.tracker_type == "train" or self.tracker_type == "holdout":
            self.loss[train_task_id] += loss.item() * total
            self.correct[train_task_id] += correct
            self.total[train_task_id] += total
            self.per_group_correct[:, train_task_id] += group_correct
            self.per_group_total[:, train_task_id] += group_total
            self.per_category_correct[:, train_task_id] += category_correct
            self.per_category_total[:, train_task_id] += category_total

            self.true_labels = np.concatenate((self.true_labels, labels_np))
            self.predicted_scores = np.vstack((self.predicted_scores, probabilities))

        else:
            self.loss[train_task_id, eval_task_id] += loss.item() * total
            self.correct[train_task_id, eval_task_id] += correct
            self.total[train_task_id, eval_task_id] += total
            self.per_group_correct[:, train_task_id, eval_task_id] += group_correct
            self.per_group_total[:, train_task_id, eval_task_id] += group_total
            self.per_category_correct[:, train_task_id, eval_task_id] += category_correct
            self.per_category_total[:, train_task_id, eval_task_id] += category_total

            self.true_labels[eval_task_id] = np.concatenate(
                (self.true_labels[eval_task_id], labels_np)
            )
            self.predicted_scores[eval_task_id] = np.vstack(
                (self.predicted_scores[eval_task_id], probabilities)
            )
