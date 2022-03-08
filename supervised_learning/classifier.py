"""
Module with class to set up a neural network classifier for casualty severity
prediction. The classifier will include the following compartments and
functionality:

- A neural network.
- A method to train the network.
- A method to make predictions for an unseen dataset. Predictions will be
returned as probabilities per class to be true (through a softmax function).
- A method to save the network.
- A method to load a network.
- A method to save and report training and test results.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from metrics import metrics_dict
import time
import numpy as np


class Classifier(object):
    """
    Artificial Neural Network classifier
    """

    def __init__(self, alpha: float, neurons: tuple, classes: int,
                 batch_size: int, input_dims: int, patience: int, cycles: int,
                 pred_batch: int, class_weights: dict = None,
                 model_name="BestModel.hdf5"):
        """
        :param alpha: Adam optimizer learning rate.
        :param neurons: Tuple with neurons per layer except from the output.
        :param classes: The number of possible classes.
        :param batch_size: Batch size for model fitting.
        :param input_dims: Network input dimension, equal to the number of
        features.
        :param patience: Training epochs to check before Early Stopping.
        :param cycles: Number of epochs to train.
        :param pred_batch: Batch size when predicting.
        :param class_weights: Class weights to be used in network fitting.
        They are saved as a dictionary {class: weight, class: weight, ...}
        :param model_name: File name to save model and weights.
        """
        self.alpha = alpha
        self.neurons = neurons
        self.classes = classes
        self.batch_size = batch_size
        self.input_dims = input_dims
        self.patience = patience
        self.cycles = cycles
        self.pred_batch = pred_batch
        self.class_weights = class_weights
        self.model_name = model_name
        self.network = self.create_ann()

    def create_ann(self):
        """
        Initialize the neural network of the classifier, based on the parameters
        defined with the class constructor method.

        :return: Artificial Neural Network (multi-layered perceptron).
        """
        ann = Sequential()
        ann.add(Dense(self.neurons[0],
                      activation="relu",
                      kernel_initializer="he_uniform",
                      input_shape=(self.input_dims,)))

        # Add intermediate layers
        for i in range(1, len(self.neurons)):
            ann.add(Dense(self.neurons[i],
                          kernel_initializer='he_uniform',
                          activation='relu'))

        # Add output layer and compile
        ann.add(Dense(self.classes, activation='softmax'))
        ann.compile(loss="sparse_categorical_crossentropy",
                    optimizer=Adam(learning_rate=self.alpha),
                    metrics=["accuracy"])
        return ann

    def fit_model(self, x_train, x_test, y_train, y_test):
        """
        :param x_train: Training samples.
        :param x_test: Test samples.
        :param y_train: Training labels.
        :param y_test: Test labels.

        :return: Model training history and runtime.
        """
        start = time.time()
        history = self.network.fit(x_train, y_train,
                                   validation_data=(x_test, y_test),
                                   epochs=self.cycles,
                                   batch_size=self.batch_size,
                                   callbacks=[
                                       EarlyStopping(verbose=True,
                                                     patience=self.patience,
                                                     monitor="val_loss"),
                                       ModelCheckpoint(self.model_name,
                                                       monitor="val_loss",
                                                       verbose=True,
                                                       save_best_only=True)
                                   ],
                                   class_weight=self.class_weights,
                                   use_multiprocessing=True)
        stop = time.time()
        runtime = round(stop - start, 3)
        runtime = time.strftime('%H:%M:%S', time.gmtime(runtime))
        return history, runtime

    def predict_probs(self, x_test):
        """
        :param x_test: Array with test data samples.

        :return: Array with predicted probabilities per class for every test
        sample.
        """
        self.network.load_weights(self.model_name)
        predictions = self.network.predict(x_test, batch_size=self.pred_batch)
        return predictions

    @staticmethod
    def classes_from_probs(predicted_probs):
        """
        :param predicted_probs: Array with rows equal to the number of
        predicted labels and columns equal to the number of classes for the
        response variable. Each row contains a predicted probability per class.

        :return: The index of the maximum probability for every row. The final
        result is the vector of predicted labels (predicted class per data
        sample).
        """
        return np.argmax(predicted_probs, axis=1)

    def predict_class_labels(self, predicted_probs: np.ndarray):
        """
        :param predicted_probs: Array with rows equal to the number of
        predicted labels and columns equal to the number of classes for the
        response variable. Each row contains a predicted probability per class.

        :return: Array with predicted class label for every test sample.
        """
        return self.classes_from_probs(predicted_probs)

    def report(self, predictions, true_labels, history, runtime):
        """
        :param predictions: Predicted casualty severity for unseen records
        with Accidents, Vehicles and Casualties information (unseen episodes).
        :param true_labels: Actual casualty severity values for the unseen
        records.
        :param history: Network fitting history.
        :param runtime: Model fitting runtime from start to Early Stopping.
        This function prints the results of the predictions for the test
        dataset y values (casualty severities).
        """
        # Get classification metrics and training accuracy
        metrics_ = metrics_dict(predictions, true_labels)
        train_accuracy = 100 * round(history.history['accuracy'][-1], 2)

        # Check epochs
        epochs_run = len(history.history['loss'])
        best_epoch = epochs_run - self.patience

        # Print results
        print(
            f"Runtime: {runtime}\n\n",
            f"Training epochs: {epochs_run}\n",
            f"Best epoch (lowest validation loss): {best_epoch}\n\n",
            f"Training accuracy: {train_accuracy} %\n",
            f"Validation accuracy: {metrics_['Validation Accuracy']} %\n\n",
            "Total correct classifications:",
            f"{metrics_['Total correct classifications']}\n",
            f"Total missed: {metrics_['Total wrong classifications']}\n\n",
            f"Confusion Matrix:\n {metrics_['Confusion Matrix']}\n\n",
            "Fatal accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][0]}\n",
            "Severe accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][1]}\n",
            "Light accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][2]}\n\n",
            f"Fatal accident class weight: {round(self.class_weights[0], 3)}\n",
            "Severe accident class weight:",
            f"{round(self.class_weights[1], 3)}\n",
            "Light accident class weight:",
            f"{round(self.class_weights[2], 3)}\n\n",
            "Average class accuracy (%):",
            f"{metrics_['Average Class Accuracy']}\n",
            "Harmonic mean of class accuracy (%):",
            f"{metrics_['Class Accuracy Harmonic Mean']}"
        )

    def save_network(self):
        """
        Save the neural network in .hdf5 format.
        """
        self.network.save(self.model_name, overwrite=True)

    def load_network(self):
        """
        Load a neural network saved in .hdf5 format.
        """
        self.network = load_model(self.model_name)
