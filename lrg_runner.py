"""
Module to run experiments with a Logistic Regression model.
"""

import pandas as pd
from supervised_learning.classifiers import LRGClassifier
from transformer import DataTransformer
from supervised_learning.class_weights import get_class_weights, CLASS_WEIGHTS

# Constants
SOLVER = "saga"
MULTICLASS = "multinomial"
PENALTY = "l2"
NAMES = ["Fatal", "Severe", "Slight"]

# Data
train_data_path = "data/data0518.feather"
test_data_path = "data/data2019.feather"


def main():
    train_data = pd.read_feather(train_data_path)
    test_data = pd.read_feather(test_data_path)

    numerical_fields = [
        'Age_of_Casualty',
        'Age_of_Driver',
        'Age_of_Vehicle',
        'Number_of_Vehicles',
        'Speed_limit'
    ]

    # Prepare data for logistic regression
    data_transformer = DataTransformer(
        train_data,
        numerical_fields,
        test_data=test_data
    )

    # Create training, validation and test sets
    x_train, x_val, y_train, y_val = data_transformer.get_train_val_data()
    x_test, y_test = data_transformer.prepare_test_data()

    # Get default class weights based on class frequency:
    default_weights = get_class_weights(y_train)
    CLASS_WEIGHTS["default"] = default_weights

    # Initialize and train model
    lrg_model = LRGClassifier(
        names=NAMES,
        penalty=PENALTY,
        class_weight=CLASS_WEIGHTS["default"],
        solver=SOLVER,
        random_state=2,
        multi_class=MULTICLASS,
        verbose=4,
        n_jobs=-1
    )
    lrg_model.fit_model(x_train, y_train)

    # Validation dataset
    y_val_predictions = lrg_model.predict(x_val)
    lrg_model.report(y_val, y_val_predictions, "VALIDATION SET RESULTS")

    # Test dataset
    y_test_predictions = lrg_model.predict(x_test)
    lrg_model.report(y_test, y_test_predictions, "TEST SET RESULTS")


if __name__ == "__main__":
    main()
