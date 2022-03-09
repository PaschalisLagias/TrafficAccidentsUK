"""
Class to transform training and test data to input format suitable for neural
networks (supervised learning and reinforcement learning).

Glossary:
- Numerical field: Data field with a continuous value.
- Categorical field: Data field that takes a finite set of values.
- Label encoder: Converts the finite distinct values of a categorical field to
integers, starting from 0.
"""
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


class DataTransformer(object):
    """
    Class to transform pre-processed data to format suitable for machine
    learning.
    """
    def __init__(self, train_data, numerical_fields, val_size: float = 0.25,
                 test_data=None):
        """
        :param train_data: Training data as pandas dataframe.
        :param numerical_fields: List of strings, the data fields to be treated
        as numerical.
        :param val_size: Percentage of the training data to be used for
        validation.
        :param test_data: Test data as pandas dataframe. If no test data is
        passed to DataTransformer, training data are split training and test
        sets.
        """
        self.train_data = train_data
        self.validation_size = val_size
        self.test_data = test_data
        self.numerical_fields = numerical_fields
        all_fields = self.train_data.columns
        non_categorical = self.numerical_fields + ["Casualty_Severity"]
        self.categorical = all_fields.symmetric_difference(non_categorical)

        self.y_label_fixer = LabelEncoder()
        self.numerical_pipe = Pipeline([("normalizer", Normalizer(copy=False))])
        self.categorical_pipe = Pipeline([
            ("one-hot encoder", OneHotEncoder(sparse=False, dtype=np.int))
        ])

        self.preprocessor = ColumnTransformer([
            ("normalizer", self.numerical_pipe, self.numerical_fields),
            ("one-hot encoder", self.categorical_pipe, self.categorical)
        ])

    def prepare_train_data(self):
        """
        Transforms input training data with the following process:
        - Numerical fields are normalized.
        - Categorical fields are one-hot encoded.
        - Casualty severity data get label-encoded.
        - Training data are split in training data and validation data (75%
        training, 25% validation).

        :return: Transformed and preprocessed training and validation X and Y.
        """
        y_train_column = self.train_data["Casualty_Severity"]
        y_train = self.y_label_fixer.fit_transform(y_train_column)
        self.train_data.drop(columns=["Casualty_Severity"], inplace=True)
        x_train = self.preprocessor.fit_transform(self.train_data)

        x_train, x_val, y_train, y_val = train_test_split(
            x_train,
            y_train,
            test_size=self.validation_size,
            shuffle=True,
            random_state=2
        )

        return x_train, x_val, y_train, y_val

    def prepare_validation_data(self):
        """
        Transforms input test data with the following process:
        - Numerical fields are normalized.
        - Categorical fields are one-hot encoded.
        - Casualty severity data get label-encoded.

        :return: Transformed and preprocessed test X and Y.
        """
        if self.test_data is not None:
            y_test_column = self.test_data["Casualty_Severity"]
            y_test = self.y_label_fixer.transform(y_test_column)
            self.test_data.drop(columns=["Casualty_Severity"], inplace=True)
            x_test = self.preprocessor.transform(self.test_data)

            return x_test, y_test
        raise AttributeError("No test dataset has been provided!")
