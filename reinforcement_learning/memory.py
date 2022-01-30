"""
Module with class to set up the replay buffer memory of the reinforcement
learning agent.

RL glossary:
- state: A training sample.
- action: Prediction of casualty severity for a training sample.
- new state: The new training sample after making a prediction.
- reward: Calculated from the result of the prediction, based on a reward
function.
- transition: The process of taking action for a state, receiving reward and
ending up to a new state.
"""

import numpy as np


class Memory(object):
    """
    Class to represent RL agent memory, where experience from episodes is saved.
    """
    def __init__(self, name_, size, input_shape, n_actions, discrete=False):
        """
        :param name_: File name to export / import memory (as .npz - numpy
        zipped array).
        :param size: Memory capacity - maximum number of records.
        :param input_shape: Number of memory columns - taken from training data.
        :param n_actions: Number of distinct possible actions.
        :param discrete: Boolean for discrete or continuous actions. For this
        project, discrete = True as the possible predicted values are 0, 1 or 2.

        Initialized variables:
        self.counter: Keeps track how many training samples have been saved in
        memory.
        self.state_memory: Stores training samples seen from the evaluation
        network.
        self.new_state_memory: Stores new states after predicting casualty
        severity for current states.
        self.action_memory: Stores action taken for current state.
        self.reward_memory: Stores reward returned by taking a step.
        self.terminal_memory: Memory to store boolean indicator for episode
        completion (0, 1).
        """
        self.name_ = name_
        self.size = size
        self.n_actions = n_actions
        self.counter = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.size, input_shape), np.float32)
        self.new_state_memory = np.zeros((self.size, input_shape), np.float32)
        datatype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.size, n_actions), dtype=datatype)
        self.reward_memory = np.zeros(size, np.float32)
        self.terminal_memory = np.zeros(self.size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        """
        :param state: Current state.
        :param action: Casualty severity prediction for current state.
        :param reward: Reward returned for action taken.
        :param state_: New state after taking action.
        :param done: Boolean for episode completion.

        Saves all components of a transition in memory.
        """
        index = self.counter % self.size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.n_actions)
            actions[action] = 1
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.counter += 1

    def sample_batch(self, batch_size):
        """
        :param batch_size: Number of samples to extract from memory.

        :return: Samples extracted from memory to train network weights.
        """
        max_memory = min(self.counter, self.size)
        batch = np.random.choice(max_memory, batch_size, replace=False)
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

# TODO: Rethink data types for memory components (np.float32 used).
