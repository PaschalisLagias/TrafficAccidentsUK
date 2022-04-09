"""
Module with function to create a deep q-network for training within a RL agent.
"""

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_dqn(l_rate, n_actions, input_dims, lyr1_dims, lyr2_dims, lyr3_dims):
    """
    :param l_rate: Learning rate for Adam optimizer.
    :param n_actions: The number of possible actions (predictions).
    :param input_dims: Network input dimension (equal to number of features)
    :param lyr1_dims: Fully connected layer 1 dimensions.
    :param lyr2_dims: Fully connected layer 2 dimensions.
    :param lyr3_dims: Fully connected layer 3 dimensions.

    :return: Returns the Q-Network to train.
    """
    # TODO: Make function more flexible with different network architectures
    model = Sequential([
        Dense(lyr1_dims, kernel_initializer='he_uniform',
              input_shape=(input_dims, )),
        Activation('relu'),
        Dense(lyr2_dims, kernel_initializer='he_uniform'),
        Activation('relu'),
        Dense(lyr3_dims,  kernel_initializer='he_uniform'),
        Activation('relu'),
        Dense(n_actions)
    ])

    model.compile(optimizer=Adam(learning_rate=l_rate), loss='mse')
    return model
