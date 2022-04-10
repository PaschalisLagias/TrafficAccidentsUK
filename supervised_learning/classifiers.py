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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from metrics import metrics_dict
import time
import numpy as np

METRICS = [
    "accuracy",
    "sparse_categorical_accuracy",
]


class ANNClassifier(object):
    """
    Artificial Neural Network classifier
    """

    def __init__(self, alpha: float, neurons: tuple, classes: int,
                 batch_size: int, input_dims: int, patience: int, cycles: int,
                 pred_batch: int, model_name="BestModel.hdf5",
                 class_weights: dict = None):
        """
        :param alpha: Adam optimizer learning rate.
        :param neurons: Tuple with neurons per hidden layer.
        :param classes: The number of possible classes.
        :param batch_size: Batch size for model fitting.
        :param input_dims: Network input dimension, equal to the number of
        features.
        :param patience: Training epochs to check before Early Stopping.
        :param cycles: Number of epochs to train.
        :param pred_batch: Batch size when predicting.
        :param model_name: File name to save model and weights.
        :param class_weights: Class weights to be used in network fitting.
        They are saved as a dictionary {class: weight, class: weight, ...}
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
        for lyr in self.neurons:
            ann.add(Dense(lyr, kernel_initializer='he_uniform',
                          activation='relu'))

        # Add output layer and compile
        ann.add(Dense(self.classes, activation='softmax'))
        ann.compile(loss="sparse_categorical_crossentropy",
                    optimizer=Adam(learning_rate=self.alpha),
                    metrics=METRICS)
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
        probabilities = self.network.predict(x_test, batch_size=self.pred_batch)
        return probabilities

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

    @staticmethod
    def best_metric(history, metric: str):
        """
        :param history: Network fitting history.
        :param metric: Metric from the metrics list passed as input during
        model compilation.

        :return: The best metric value recorded during model training.
        """
        best = max(history.history[metric])
        return 100 * round(best, 2)

    def report(self, true_labels, predictions, history, runtime):
        """
        :param true_labels: Actual casualty severity values for the unseen
        records.
        :param predictions: Predicted casualty severity for unseen records
        with Accidents, Vehicles and Casualties information (unseen episodes).
        :param history: Network fitting history.
        :param runtime: Model fitting runtime from start to Early Stopping.

        This function prints the results of the predictions for the test
        dataset y values (casualty severities).
        """
        # Get classification metrics and training accuracy
        metrics_ = metrics_dict(true_labels, predictions)
        train_accuracy = self.best_metric(history, "accuracy")
        train_sparse_cat_accuracy = self.best_metric(
            history,
            "sparse_categorical_accuracy"
        )
        val_accuracy = self.best_metric(history, "val_accuracy")
        val_sparse_cat_accuracy = self.best_metric(
            history,
            "val_sparse_categorical_accuracy"
        )

        # Check epochs
        epochs_run = len(history.history['loss'])
        best_epoch = epochs_run - self.patience

        # Print results
        print(
            f"Runtime: {runtime}\n\n",
            f"Training epochs: {epochs_run}\n",
            f"Best epoch (lowest validation loss): {best_epoch}\n\n",
            f"Training accuracy: {train_accuracy} %\n",
            f"Validation accuracy: {val_accuracy} %\n",
            f"Training Sparse Categorical Accuracy:",
            f"{train_sparse_cat_accuracy} %\n",
            f"Validation Sparse Categorical Accuracy:",
            f"{val_sparse_cat_accuracy} %\n",
            f"Test accuracy: {metrics_['Accuracy']} %\n\n",
            "Total correct classifications:",
            f"{metrics_['Total correct classifications']}\n",
            f"Total missed: {metrics_['Total wrong classifications']}\n\n",
            f"Confusion Matrix:\n {metrics_['Confusion Matrix']}\n\n",
            "Fatal accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][0]}\n",
            "Severe accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][1]}\n",
            "Light accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][2]}\n\n"
        )

        # Print class weights if not None
        if self.class_weights:
            print(
                f"Fatal accident class weight:",
                f"{round(self.class_weights[0], 3)}\n",
                "Severe accident class weight:",
                f"{round(self.class_weights[1], 3)}\n",
                "Light accident class weight:",
                f"{round(self.class_weights[2], 3)}\n\n"
            )

        print(
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


class LRGClassifier(LogisticRegression):
    """
    A Logistic Regression Classifier. Apart from the default variables from
    scikit-learn Logistic Regression class, methods are added for timed model
    fitting and reporting of respective results.
    """
    def __init__(self, names=None, penalty='l2', dual=False, tol=0.0001, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):
        """
        :param names: Casualty severity class names.
        Please refer to sklearn.linear_model.LogisticRegression documentation
        for detailed information about model parameters:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?msclkid=1c37939db8bd11ec89f5dbf1fe324f9d
        """
        super(LRGClassifier, self).__init__(
            penalty=penalty,
            dual=dual,
            tol=tol,
            C=C,
            fit_intercept=fit_intercept,
            intercept_scaling=intercept_scaling,
            class_weight=class_weight,
            random_state=random_state,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            verbose=verbose,
            warm_start=warm_start,
            n_jobs=n_jobs,
            l1_ratio=l1_ratio
        )
        self.runtime = None
        self.names = names

    def fit_model(self, x_train, y_train):
        """
        Fit the logistic regression model to the training data. Model fitting
        is timed.
        """
        start = time.time()
        self.fit(x_train, y_train)
        stop = time.time()
        runtime = round(stop - start, 3)
        self.runtime = time.strftime('%H:%M:%S', time.gmtime(runtime))

    def report(self, true_labels, predictions, title: str):
        """
        :param true_labels: Actual casualty severity values for the unseen
        records.
        :param predictions: Predicted casualty severity for unseen records
        with Accidents, Vehicles and Casualties information (unseen episodes).
        :param title: Report title e.g. 'VALIDATION SET RESULTS'.

        This function reports the results of predictions for a dataset (e.g. a
        validation or a test dataset).
        """
        metrics_ = metrics_dict(true_labels, predictions)
        results = classification_report(true_labels, predictions,
                                        target_names=self.names)

        # Print metrics and results
        print(
            f"{title}",
            f"Runtime: {self.runtime}\n\n",
            f"Accuracy: {metrics_['Accuracy']} %\n\n",
            "Total correct classifications:",
            f"{metrics_['Total correct classifications']}\n",
            f"Total missed: {metrics_['Total wrong classifications']}\n\n",
            f"Confusion Matrix:\n {metrics_['Confusion Matrix']}\n\n",
            "Fatal accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][0]}\n",
            "Severe accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][1]}\n",
            "Light accident classification accuracy (%):",
            f"{metrics_['Class Accuracies'][2]}\n\n"
        )

        # Print class weights if not None
        class_weights = self.get_params()["class_weight"]
        if class_weights:
            print(
                f"Fatal accident class weight: {round(class_weights[0], 3)}\n",
                f"Severe accident class weight: {round(class_weights[1], 3)}\n",
                f"Light accident class weight: {round(class_weights[2], 3)}\n\n"
            )

        print(
            "Average class accuracy (%):",
            f"{metrics_['Average Class Accuracy']}\n",
            "Harmonic mean of class accuracy (%):",
            f"{metrics_['Class Accuracy Harmonic Mean']}\n\n",
            f"Classification Report:\n\n{results}"
        )
