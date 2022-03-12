"""
Short module with function to compute dictionary of class weights to be used
while training a neural network. Class weights are applied at loss function
stage during training, making the model more sensitive towards the class with
the largest weight.
"""

from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def get_class_weights(y_train):
    """
    :param y_train: Training data labels.
    :return: Class weights to be used in model training.
    """
    counts = compute_class_weight(class_weight="balanced",
                                  classes=np.unique(y_train),
                                  y=y_train)
    weights = {i: w for i, w in enumerate(counts)}
    return weights
