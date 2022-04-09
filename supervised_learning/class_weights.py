"""
Short module with function to compute dictionary of class weights to be used
while training a neural network. Class weights are applied at loss function
stage during training, making the model more sensitive towards the class with
the largest weight.
"""

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Dictionary with class weights for neural network ensembles
CLASS_WEIGHTS = {
    "fatal": {0: 26.5, 1: 1.44, 2: 0.32},
    "severe": {0: 16.5, 1: 4.68, 2: 0.62},
    "average": {0: 19.5, 1: 3.68, 2: 0.62}
}


def get_class_weights(y_train: np.ndarray):
    """
    :param y_train: Training data labels.

    :return: Dictionary with class weights to be used in model training. Class
    labels are the keys and weights are the values of the dictionary.
    """
    counts = compute_class_weight(class_weight="balanced",
                                  classes=np.unique(y_train),
                                  y=y_train)
    weights = {i: w for i, w in enumerate(counts)}
    return weights
