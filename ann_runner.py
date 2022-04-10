"""
Module to run supervised learning experiments.
"""

import pandas as pd
from supervised_learning.majority_vote import average_classes
from supervised_learning.classifiers import ANNClassifier
from metrics import metrics_dict
from transformer import DataTransformer
from supervised_learning.class_weights import get_class_weights, CLASS_WEIGHTS

# Constants
BATCH_SIZE = 512
LEARNING_RATE = 0.001
NUMBER_OF_CLASSES = 3
STOPPING_CHECKS = 10
TRAINING_EPOCHS = 200

train_data_path = "data/data0518.feather.feather"
test_data_path = "data/data2019.feather"


def main():
    """
    Function that runs the entire supervised learning process with an artificial
    neural network classifier.
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
    data_transformer = DataTransformer(train_data, numerical_fields,
                                       test_data=test_data)

    # Create training, validation and test sets
    x_train, x_val, y_train, y_val = data_transformer.get_train_val_data()
    x_test, y_test = data_transformer.prepare_test_data()
    inp_shape = x_train.shape[1]

    # Get default class weights based on class frequency:
    default_weights = get_class_weights(y_train)
    CLASS_WEIGHTS["default"] = default_weights

    # CREATE MODELS
    # Neural network sensitive to fatal casualty severity
    ann_fatal = ANNClassifier(LEARNING_RATE, (100, 1000, 1000),
                              NUMBER_OF_CLASSES,
                              BATCH_SIZE, inp_shape, STOPPING_CHECKS,
                              TRAINING_EPOCHS, BATCH_SIZE,
                              "BestModel_for_FatalCasualties.hdf5",
                              CLASS_WEIGHTS["fatal"]
                              )

    # Neural network sensitive to severe casualty severity
    ann_severe = ANNClassifier(LEARNING_RATE, (100, 800, 1000),
                               NUMBER_OF_CLASSES,
                               BATCH_SIZE, inp_shape, STOPPING_CHECKS,
                               TRAINING_EPOCHS, BATCH_SIZE,
                               "BestModel_for_SevereCasualties.hdf5",
                               CLASS_WEIGHTS["severe"]
                               )

    # Neural network skilled at getting good average class accuracy
    ann_avg = ANNClassifier(LEARNING_RATE, (2000, 500), NUMBER_OF_CLASSES,
                            BATCH_SIZE, inp_shape, STOPPING_CHECKS,
                            TRAINING_EPOCHS, BATCH_SIZE,
                            "BestModel_for_AverageClassAccuracy.hdf5",
                            CLASS_WEIGHTS["average"]
                            )

    # FIT MODELS AND GET CLASS PROBABILITIES AND PREDICTIONS PER MODEL
    # Neural network skilled at finding fatal injuries
    print("FITTING MODEL SENSITIVE TO FATAL CASUALTIES...\n")
    ann_fatal_fit, ann_fatal_timer = ann_fatal.fit_model(
        x_train,
        x_val,
        y_train,
        y_val
    )

    fatal_ann_probs = ann_fatal.predict_probs(x_test)
    fatal_ann_predictions = ann_fatal.predict_class_labels(fatal_ann_probs)

    # Neural network skilled at finding severe injuries
    print("\n\n\nFITTING MODEL SENSITIVE TO SEVERE CASUALTIES...\n")
    ann_severe_fit, ann_severe_timer = ann_severe.fit_model(
        x_train,
        x_val,
        y_train,
        y_val
    )

    ann_severe_probs = ann_severe.predict_probs(x_test)
    ann_severe_predictions = ann_severe.predict_class_labels(ann_severe_probs)

    # Neural network skilled at getting good average class accuracy
    print("\n\n\nFITTING MODEL TO GET GOOD AVERAGE CLASS ACCURACY...\n\n")
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
    print("\nREPORT RESULTS FOR FATAL CASUALTY SENSITIVE MODEL:\n")
    ann_fatal.report(
        y_test,
        fatal_ann_predictions,
        ann_fatal_fit,
        ann_fatal_timer
    )

    # Neural network sensitive to severe injuries
    print("\n\n\nREPORT RESULTS FOR SEVERE CASUALTY SENSITIVE MODEL:\n")
    ann_severe.report(
        y_test,
        ann_severe_predictions,
        ann_severe_fit,
        ann_severe_timer
    )

    # Neural network with good average class accuracy
    print("\n\n\nREPORT RESULTS FOR MODEL WITH GOOD AVERAGE CLASS ACCURACY:\n")
    ann_avg.report(
        y_test,
        ann_avg_predictions,
        ann_avg_fit,
        ann_avg_timer
    )

    # MAJORITY VOTE
    ensemble_predictions = average_classes([
        fatal_ann_probs,
        ann_severe_probs,
        ann_avg_probs
    ])

    # Get metrics for ensemble predictions
    avg_metrics = metrics_dict(y_test, ensemble_predictions)

    # Print results
    print(
        "\n\nPRINT RESULTS AFTER MAJORITY VOTE:\n",
        f"Test accuracy: {avg_metrics['Accuracy']} %\n\n",
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
