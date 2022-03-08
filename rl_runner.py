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
train_data_path = None
test_data_path = None


def main():
    """
    Function that runs the entire reinforcement learning process.
    """
    mode = MODE
    train_data = pd.read_feather(train_data_path)
    test_data = pd.read_feather(test_data_path)
    numerical_fields = [
        'Age_of_Casualty',
        'Age_of_Driver',
        'Age_of_Vehicle',
        'Number_of_Vehicles',
        'Speed_limit'
    ]
    data_transformer = DataTransformer(train_data, numerical_fields, test_data)
    x_train, x_test, y_train, y_test = data_transformer.prepare_data()
    inp_shape = x_train.shape[1]
    environment = Accidents(x_train, y_train)
    if mode == "start":
        epsilon_initial = EPSILON_INITIAL
    else:
        epsilon_initial = EPSILON_FINAL
    agent = Agent(alpha=LEARNING_RATE, gamma=GAMMA, n_actions=3,
                  epsilon=epsilon_initial, batch_size=BATCH_SIZE,
                  input_dims=inp_shape, epsilon_dec=EPSILON_DECREMENT,
                  epsilon_end=EPSILON_FINAL, mem_size=MEMORY_SIZE)

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

            # Save transition in memory:
            agent.remember(observation, action, reward, observation_, done)
            observation = observation_

            # Learn and update Q-Network:
            agent.learn()

        print(f"Epsilon: {round(agent.epsilon, 3)}")

    # Make predictions and report results:
    predictions = agent.make_predictions(test_episodes=x_test, batch_size=400)
    agent.report(predictions, y_test)
    agent.save_model()
    agent.save_memory()


if __name__ == "__main__":
    main()
