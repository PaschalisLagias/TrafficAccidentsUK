"""
Module to run supervised learning experiments.
"""

import pandas as pd
from supervised_learning.majority_vote import average_classes
from supervised_learning.classifier import Classifier
from metrics import metrics_dict
from transformer import DataTransformer

# constants
BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUMBER_OF_CLASSES = 3
STOPPING_CHECKS = 5
TRAINING_EPOCHS = 2
CLASS_WEIGHTS = {
    "fatal": {0: 17.5, 1: 2.44, 2: 0.69},
    "severe": {0: 19.5, 1: 3.44, 2: 0.69},
    "average": {0: 31.83, 1: 3.04, 2: 0.38}
}
train_data_path = "data/data0518.feather.feather"
test_data_path = "data/data2019.feather"


def main():
    """
    Function that runs the entire supervised learning process.
    """
    train_data = pd.read_feather(train_data_path)
    test_data = pd.read_feather(test_data_path)

    numerical_fields = [
        'Age_of_Casualty',
        'Age_of_Driver',
        'Age_of_Vehicle',
        'Number_of_Vehicles',
        'Speed_limit'
    ]

    # Prepare data for neural network models
    data_transformer = DataTransformer(
        train_data,
        numerical_fields,
        0.25,
        test_data
    )
    x_train, x_val, y_train, y_val = data_transformer.prepare_train_data()
    x_test, y_test = data_transformer.prepare_validation_data()
    inp_shape = x_train.shape[1]

    # CREATE MODELS
    # Neural network sensitive to fatal casualty severity
    ann_fatal = Classifier(LEARNING_RATE, (1000, 1000, 1000), NUMBER_OF_CLASSES,
                           BATCH_SIZE, inp_shape, STOPPING_CHECKS,
                           TRAINING_EPOCHS, BATCH_SIZE,
                           "BestModel_for_FatalCasualties.hdf5",
                           CLASS_WEIGHTS["fatal"]
                           )

    # Neural network sensitive to severe casualty severity
    ann_severe = Classifier(LEARNING_RATE, (2000, 500), NUMBER_OF_CLASSES,
                            BATCH_SIZE, inp_shape, STOPPING_CHECKS,
                            TRAINING_EPOCHS, BATCH_SIZE,
                            "BestModel_for_SevereCasualties.hdf5",
                            CLASS_WEIGHTS["severe"]
                            )

    # Neural network without class weights
    ann_ = Classifier(LEARNING_RATE, (1200, 1200, 1200), NUMBER_OF_CLASSES,
                      BATCH_SIZE, inp_shape, STOPPING_CHECKS,
                      TRAINING_EPOCHS, BATCH_SIZE,
                      "BestModel_for_ValidationAccuracy.hdf5"
                      )

    # Neural network skilled at getting good average class accuracy
    ann_avg = Classifier(LEARNING_RATE, (1200, 1200, 1200), NUMBER_OF_CLASSES,
                         BATCH_SIZE, inp_shape, STOPPING_CHECKS,
                         TRAINING_EPOCHS, BATCH_SIZE,
                         "BestModel_for_AverageClassAccuracy.hdf5",
                         CLASS_WEIGHTS["average"]
                         )

    # FIT MODELS AND GET CLASS PROBABILITIES AND PREDICTIONS PER MODEL
    # Neural network skilled at finding fatal injuries
    print("Fitting model sensitive to fatal casualties...\n")
    ann_fatal_fit, ann_fatal_timer = ann_fatal.fit_model(
        x_train,
        x_val,
        y_train,
        y_val
    )

    fatal_ann_probs = ann_fatal.predict_probs(x_test)
    fatal_ann_predictions = ann_fatal.predict_class_labels(fatal_ann_probs)

    # Neural network skilled at finding severe injuries
    print("\n\n\nFitting model sensitive to severe casualties...\n")
    ann_severe_fit, ann_severe_timer = ann_severe.fit_model(
        x_train,
        x_val,
        y_train,
        y_val
    )

    ann_severe_probs = ann_severe.predict_probs(x_test)
    ann_severe_predictions = ann_severe.predict_class_labels(ann_severe_probs)

    # Neural network without class weights
    print("\n\n\nFitting model to get good average class accuracy...\n\n")
    ann_fit, ann_timer = ann_.fit_model(
        x_train,
        x_val,
        y_train,
        y_val
    )

    ann_probs = ann_avg.predict_probs(x_test)
    ann_predictions = ann_.predict_class_labels(ann_probs)

    # Neural network skilled at getting good average class accuracy
    print("\n\n\nFitting model to get good average class accuracy...\n\n")
    ann_avg_fit, ann_avg_timer = ann_avg.fit_model(
        x_train,
        x_val,
        y_train,
        y_val
    )

    ann_avg_probs = ann_avg.predict_probs(x_test)
    ann_avg_predictions = ann_avg.predict_class_labels(ann_avg_probs)

    # REPORT RESULTS
    # Neural network sensitive to fatal injuries
    print("\nReport results for fatal casualty sensitive model:\n")
    ann_fatal.report(
        fatal_ann_predictions,
        y_test,
        ann_fatal_fit,
        ann_fatal_timer
    )

    # Neural network sensitive to severe injuries
    print("\n\n\nReport results for severe casualty sensitive model:\n")
    ann_severe.report(
        ann_severe_predictions,
        y_test,
        ann_severe_fit,
        ann_severe_timer
    )

    # Neural network without class weights
    print("\n\n\nReport results for model with good average class accuracy:\n")
    ann_.report(
        ann_predictions,
        y_test,
        ann_fit,
        ann_timer
    )

    # Neural network with good average class accuracy
    print("\n\n\nReport results for model with good average class accuracy:\n")
    ann_avg.report(
        ann_avg_predictions,
        y_test,
        ann_avg_fit,
        ann_avg_timer
    )

    # Get majority vote
    avg_predictions = average_classes([
        fatal_ann_probs,
        ann_severe_probs,
        ann_probs,
        ann_avg_probs
    ])

    # Get metrics for ensemble predictions
    avg_metrics = metrics_dict(avg_predictions, y_test)

    # Print results
    print(
        "\n\nPrint results after majority vote:\n",
        f"Validation accuracy: {avg_metrics['Validation Accuracy']} %\n\n",
        "Total correct classifications:",
        f"{avg_metrics['Total correct classifications']}\n",
        f"Total missed: {avg_metrics['Total wrong classifications']}\n\n",
        f"Confusion Matrix:\n {avg_metrics['Confusion Matrix']}\n\n",
        "Fatal accident classification accuracy (%):",
        f"{avg_metrics['Class Accuracies'][0]}\n",
        "Severe accident classification accuracy (%):",
        f"{avg_metrics['Class Accuracies'][1]}\n",
        "Light accident classification accuracy (%):",
        f"{avg_metrics['Class Accuracies'][2]}\n\n",
        "Average class accuracy (%):",
        f"{avg_metrics['Average Class Accuracy']}\n",
        "Harmonic mean of class accuracy (%):",
        f"{avg_metrics['Class Accuracy Harmonic Mean']}"
    )


if __name__ == "__main__":
    main()
