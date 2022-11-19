"""
Module to run reinforcement learning experiments.
"""

import pandas as pd

from transformer import DataTransformer
from reinforcement_learning.environment import Accidents
from reinforcement_learning.agent import Agent

# constants
GAMMA = 0.1
BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPSILON_DECREMENT = 0.998
EPSILON_INITIAL = 1.0
EPSILON_FINAL = 0.01
MEMORY_SIZE = 800000
TRAIN_EPISODES = 50000
REWARD_STYLE = "reversed"
PENALTY_STYLE = "standard"
MODE = "Start"
DQN_NAME = "doubleQNet.h5"
MEMORY_NAME = "double_qnet_memory.npz"
CLASS_NAMES = ["Fatal", "Severe", "Slight"]
train_data_path = "data/data0518.feather"
test_data_path = "data/data2019.feather"


def main():
    """
    Function that runs the entire reinforcement learning process.
    """
    mode = MODE
    numerical_fields = [
        'Age_of_Casualty',
        'Age_of_Driver',
        'Age_of_Vehicle',
        'Number_of_Vehicles',
        'Speed_limit'
    ]
    train_data = pd.read_feather(train_data_path)
    test_data = pd.read_feather(test_data_path)

    # Create training, validation and test sets
    data_transformer = DataTransformer(train_data, numerical_fields,
                                       test_data=test_data)
    x_train, x_val, y_train, y_val = data_transformer.get_train_val_data()
    x_test, y_test = data_transformer.prepare_test_data()

    # Prepare environment and agent
    environment = Accidents(x_train, y_train)
    inp_shape = x_train.shape[1]
    if mode == "start":
        epsilon_initial = EPSILON_INITIAL
    else:
        epsilon_initial = EPSILON_FINAL
    agent = Agent(alpha=LEARNING_RATE, gamma=GAMMA, n_actions=3,
                  names=CLASS_NAMES, epsilon=epsilon_initial,
                  batch_size=BATCH_SIZE, input_dims=inp_shape,
                  epsilon_dec=EPSILON_DECREMENT, epsilon_end=EPSILON_FINAL,
                  mem_size=MEMORY_SIZE, dqn_name=DQN_NAME,
                  mem_name=MEMORY_NAME, replace_target=500)

    # Load model weights and memory if training is restarted:
    if mode == "continue":
        agent.load_model()
        agent.load_memory()
    observation = environment.x[0]

    # Start running episodes:
    for i in range(TRAIN_EPISODES):
        done = False

        # Keep choosing actions and taking steps while the episode is not over:
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = environment.step(action)

            # Save transition in memory, learn and update Q-network:
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()

        print(f"Epsilon: {round(agent.epsilon, 3)}")

    # Make predictions, report results and save model:
    print("REPORT VALIDATION SET RESULTS\n")
    val_predictions = agent.make_predictions(x_val, BATCH_SIZE)
    agent.report(y_val, val_predictions)

    print("\n\nREPORT TEST SET RESULTS\n")
    test_predictions = agent.make_predictions(x_test, BATCH_SIZE)
    agent.report(y_test, test_predictions)

    agent.save_model()
    agent.save_memory()


if __name__ == "__main__":
    main()
