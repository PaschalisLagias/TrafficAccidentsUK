"""
Function to get the majority vote from a group of neural networks by averaging
their class probability predictions and finding the class per data sample using
majority vote.
"""

import numpy as np


def average_classes(all_predictions: list):
    """
    :param all_predictions: List of nupy arrays, where each array holds the
    probability predictions for every possible class of the response
    variable. All arrays in the list will be of equal size.

    :return: Array with element-wise means of the arrays in the list. This
    method is used to average the predicted class probabilities from a group
    of neural networks.
    """
    probability_sums = np.sum(all_predictions, axis=0)
    num_of_networks = len(all_predictions)
    averaged_arr = np.divide(probability_sums, num_of_networks)
    return np.argmax(averaged_arr, axis=1)
