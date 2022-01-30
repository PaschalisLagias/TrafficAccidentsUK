"""
Class to transform training and test data to input format suitable for neural
networks (supervised learning and reinforcement learning).
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
    def __init__(self, train_data, numerical_fields, test_data=None):
        """
        :param train_data: Training data as pandas dataframe.
        :param numerical_fields: List of strings, the fields to be treated as
        numerical in training data.
        :param test_data: Test data as pandas dataframe. If no test data is
        passed to the class, DataTransformer splits training data in training
        and test sets.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.numerical_fields = numerical_fields
        all_fields = self.train_data.columns
        non_categorical = self.numerical_fields + ["Casualty_Severity"]
        self.categorical = all_fields.symmetric_difference(non_categorical)
        self.numerical_pipe = Pipeline([
            ("normalization", Normalizer(copy=False))
        ])
        self.categorical_pipe = Pipeline([
            ("one hot encoding", OneHotEncoder(sparse=False, dtype=np.int))
        ])
        self.label_fixer = LabelEncoder()
        self.preprocessor = ColumnTransformer([
            ("normalization", self.numerical_pipe, self.numerical_fields),
            ("one hot encoding", self.categorical_pipe, self.categorical)
        ])

    def prepare_data(self):
        """
        Transforms input training and test data with the following process:
        - Numerical fields are normalized.
        - Categorical fields are one-hot encoded.
        - Casualty severity data get label-encoded.
        - If no test dataset is provided, training data are split to training
        and test datasets.

        :return: Transformed and preprocessed training and test X and Y.
        """
        train_response = self.train_data["Casualty_Severity"]
        y_train = self.label_fixer.fit_transform(train_response)
        self.train_data.drop(columns=["Casualty_Severity"], inplace=True)
        x_train = self.preprocessor.fit_transform(self.train_data)

        if self.test_data is not None:
            test_response = self.test_data["Casualty_Severity"]
            y_test = self.label_fixer.transform(test_response)
            self.test_data.drop(columns=["Casualty_Severity"], inplace=True)
            x_test = self.preprocessor.transform(self.test_data)
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x_train,
                y_train,
                shuffle=True,
                random_state=2
            )
        return x_train, x_test, y_train, y_test
