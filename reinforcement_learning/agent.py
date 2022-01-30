"""
Module with class to set up the reinforcement learning, which will explore the
Accidents environment and use Memory to train its deep Q network.

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

from memory import Memory
from dqnetwork import create_dqn

from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix


class Agent(object):
    """
    Agent class to explore the Accidents environment and run reinforcement
    learning experiments.
    """

    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.996, epsilon_end=0.01,
                 mem_size=10000, dqn_name="doubleQNet.h5",
                 mem_name="dqn_memory.npz", replace_target=500):
        """
        :param alpha: Adam optimizer learning rate.
        :param gamma: Reward discount parameter.
        :param n_actions: Number of distinct actions.
        :param epsilon: Initial epsilon, used for epsilon-greedy policy.
        :param batch_size: Number of records to sample from memory.
        :param input_dims: Number of fields in a training sample.
        :param epsilon_dec: Epsilon decrement for epsilon-greedy policy.
        :param epsilon_end: Minimum epsilon value.
        :param mem_size: Memory size (max number of records to store).
        :param dqn_name: File name to save target network.
        :param mem_name: File name to save memory.
        :param replace_target: Number of episodes played before updating target
        network.

        Initializes Agent parameters, memory (instance of Memory class),
        evaluation Q network and target Q network.
        """
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.dqn_name = dqn_name
        self.replace_target = replace_target
        self.memory = Memory(mem_name, mem_size, input_dims, n_actions,
                             discrete=True)
        self.eval_dqn = create_dqn(alpha, n_actions, input_dims,
                                   1200, 1200, 1200)
        self.target_dqn = create_dqn(alpha, n_actions, input_dims,
                                     1200, 1200, 1200)

    def remember(self, state, action, reward, new_state, done):
        """
        :param state: Initial state.
        :param action: Action taken.
        :param reward: Reward returned from action.
        :param new_state: New state after action.
        :param done: Episode termination check (boolean flag).

        Stores a transition in agent memory.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        """
        Takes action for the current state based on the epsilon greedy policy.

        :param state: Current state.

        :return: Action: Action taken for the current state.
        """
        state = state[np.newaxis, :]
        rand_num = np.random.random()
        if rand_num < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.eval_dqn.predict(state)
            action = np.argmax(actions)
        return action

    def learn(self):
        """
        Function to learn from the data.

        Process summary:
        - Checks if there is at least one full batch of transitions stored in
        memory.
        - Samples a batch of stored transitions from memory. Each transition
        includes: state, action taken, reward, new state and done flag.
        -
        """
        if self.memory.counter < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_batch(
            self.batch_size
        )
        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(action, action_values)
        q_current = self.eval_dqn.predict(state)
        q_next = self.target_dqn.predict(new_state)
        q_eval = self.eval_dqn.predict(new_state)
        max_actions = np.argmax(q_eval, axis=1)
        q_target = q_current
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        q_target[batch_index, action_indices] = reward + \
                self.gamma * q_next[batch_index, max_actions.astype(int)] * done
        _ = self.eval_dqn.fit(state, q_target, verbose=0,
                              use_multiprocessing=True)
        if self.epsilon > self.epsilon_end:
            self.epsilon = self.epsilon * self.epsilon_dec
        else:
            self.epsilon = self.epsilon_end
        if self.memory.counter % self.replace_target == 0:
            self.update_net_weights()

    def update_net_weights(self):
        """Update target network weights."""
        self.target_dqn.set_weights(self.eval_dqn.get_weights())

    def save_model(self):
        """Save the model in a file."""
        self.eval_dqn.save(self.dqn_name, overwrite=True)

    def save_memory(self):
        """Save agent memory as a .npz file for retraining."""
        counter_memory = np.array([self.memory.counter, 0])
        np.savez_compressed(
            self.memory.name_,
            counter_memory,
            self.memory.state_memory,
            self.memory.new_state_memory,
            self.memory.action_memory,
            self.memory.reward_memory,
            self.memory.terminal_memory
        )

    def load_model(self):
        """Load a model from file."""
        self.eval_dqn = load_model(self.dqn_name)
        if self.epsilon <= self.epsilon_end:
            self.update_net_weights()

    def load_memory(self):
        """Loads memory to agent from .npz file for further training."""
        memory = np.load(self.memory.name_)
        self.memory.counter = memory["arr_0"][0]
        self.memory.state_memory = memory["arr_1"]
        self.memory.new_state_memory = memory["arr_2"]
        self.memory.action_memory = memory["arr_3"]
        self.memory.reward_memory = memory["arr_4"]
        self.memory.terminal_memory = memory["arr_5"]

    def make_predictions(self, test_episodes, batch_size):
        """
        Makes predictions for a batch of unseen data samples.

        :param test_episodes: Numpy array with unseen training samples (test X).
        :param batch_size: Batch size.

        :return: Casualty severity predictions for the test data.
        """
        self.target_dqn.add(Activation("softmax"))
        predictions = self.target_dqn.predict_classes(
            test_episodes,
            batch_size=batch_size
        )
        return predictions

    @staticmethod
    def report(predictions, true_labels):
        """
        Prints the results of the predictions compared with the test y labels.

        :param predictions: Predicted casualty severity for test data records.
        :param true_labels: Actual casualty severity for test data records.
        """
        conf_matrix = confusion_matrix(true_labels, predictions)
        true_pos = np.diag(conf_matrix)
        true_cases = conf_matrix.sum(axis=1)
        class_acc = np.multiply(np.divide(true_pos, true_cases), 100)
        class_acc = np.around(class_acc, decimals=3)
        avg_accuracy = np.around(np.mean(class_acc), decimals=3)
        harm_mean = np.around(np.divide(3, np.sum(np.divide(1, class_acc))), 3)
        total_found = sum(true_pos)
        test_size = true_labels.shape[0]
        total_missed = test_size - total_found
        val_accuracy = round(100 * (total_found / test_size), 2)

        print(f"\nValidation accuracy: {val_accuracy} %\n\n",
              f"Total correct classifications: {total_found}\n",
              f"Total missed: {total_missed}\n\n",
              f"Confusion Matrix:\n{conf_matrix}\n\n",
              f"Fatal accident classification accuracy (%): {class_acc[0]}\n",
              f"Severe accident classification accuracy (%): {class_acc[1]}\n",
              f"Light accident classification accuracy (%): {class_acc[2]}\n\n",
              f"Average class accuracy (%): {avg_accuracy}\n",
              f"Harmonic mean of class accuracy (%): {harm_mean}")
