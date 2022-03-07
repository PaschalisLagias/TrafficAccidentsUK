"""
Module with function to calculate classification metrics. The function is
written to assist reporting for classification results from various classifiers
e.g. the reinforcement learning agent or the supervised learning classifier.
"""

from sklearn.metrics import confusion_matrix
import numpy as np


def metrics_dict(predictions: np.ndarray, true_labels: np.ndarray):
    """
    :param predictions: Array with predicted casualty severity class labels.
    :param true_labels: Array with actual casualty severity class labels.

    :return: Dictionary with classification metrics
    """
    # Calculate confusion matrix, true positives and true cases
    conf_matrix = confusion_matrix(true_labels, predictions)
    true_pos = np.diag(conf_matrix)
    true_cases = conf_matrix.sum(axis=1)

    # Calculate individual class accuracy (%)
    class_acc = np.multiply(np.divide(true_pos, true_cases), 100)
    class_acc = np.around(class_acc, decimals=3)

    # Harmonic mean and average of class accuracy
    avg_accuracy = np.around(np.mean(class_acc), decimals=3)
    harm_mean = np.around(np.divide(3, np.sum(np.divide(1, class_acc))), 3)

    # Validation (or test) accuracy
    total_found = sum(true_pos)
    test_size = true_labels.shape[0]
    total_missed = test_size - total_found
    val_accuracy = round(100 * (total_found / test_size), 2)

    # Save metrics in a dictionary
    metrics = {
        "Confusion Matrix": conf_matrix,
        "Average Class Accuracy": avg_accuracy,
        "Class Accuracy Harmonic Mean": harm_mean,
        "Class Accuracies": class_acc,
        "Validation Accuracy": val_accuracy,
        "Total correct classifications": total_found,
        "Total wrong classifications": total_missed,
        "Number of test data samples": test_size
    }

    return metrics



