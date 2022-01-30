"""
Module with class to set up the environment for reinforcement learning. The
class inherits from gym.Env class.

RL glossary:
- state: A training sample.
- action: Prediction of casualty severity for a training sample.
- new state: The new training sample after making a prediction.
- reward: Calculated from the result of the prediction, based on a reward
function.
- transition: The process of taking action for a state, receiving reward and
ending up to a new state.
"""

import gym
import numpy as np
from gym import spaces
from collections import Counter
from reinforcement_learning import rewards


class Accidents(gym.Env):
    """
    Custom environment class, which inherits from gym Env base class.
    """
    def __init__(self, x, y, reward_style="reversed", penalty_style="standard"):
        """
        :param x: Training observations as numpy array.
        :param y: Training labels as numpy array.
        :param reward_style: String indicating which reward function to use
        from the imported module rewards to compute reward for prediction per
        severity type.
        :param penalty_style: String indicating which function to use from
        rewards imported module to calculate rewards and penalties.

        Initialized variables:
        self.samples: Number of training data records.
        Self.variables: Number of training data columns.
        self.casualty_sev_dict: Dictionary mapping casualty severity values to
        their actual meaning.
        self.action_space = Number of distinct possible actions (predictions).
        self.step_counter: Counter to track the number of predictions (steps
        taken) within every single episode.
        self.episode_counter: Counter to track how many episodes have been
        played.
        self.current_state: The training sample that needs prediction for
        casualty severity.
        self.total_reward: Variable to keep track of the reward collected
        within one entire episode.
        self.reward_style: Option to calculate reward per severity type.
        self.penalty_style: Option to calculate reward / penalty based on
        prediction result (correct / wrong).
        self.rewards_: Dictionary that maps casualty severity type values to
        their respective rewards.
        """
        super().__init__()
        self.x = x
        self.y = y
        self.samples = x.shape[0]
        self.variables = x.shape[1]
        self.casualty_sev_dict = {0: "Fatal", 1: "Severe", 2: "Light"}
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(1, x.shape[1])
        )
        self.step_counter = 0
        self.episode_counter = 0
        self.current_state = x[0]
        self.total_reward = 0
        self.reward_style = reward_style
        self.penalty_style = penalty_style
        self.rewards_ = self.get_reward_dict()

    def get_reward_dict(self):
        """
        :return: Dictionary with severity types (integers) as keys and their
        respective rewards as values.

        Instance attribute self.reward_style defines which function to use from
        rewards imported module.
        An incorrect entry for self.reward_style attribute raises a ValueError.
        """
        counts = Counter(self.y)
        if self.reward_style == "reversed":
            self.rewards_ = rewards.prop_rev_rewards(counts)
        elif self.reward_style == "from minority":
            self.rewards_ = rewards.rewards_from_minority(counts)
        else:
            raise ValueError("Wrong value for reward style")

    def shuffle_data(self):
        """
        Reshuffles training data and training labels to reset the environment
        and start a new episode.
        """
        indexes = np.random.rand(self.samples).argsort()
        self.x = np.take(self.x, indexes, axis=0)
        self.y = np.take(self.y, indexes, axis=0)

    def reset(self):
        """
        Reset the environment and draw the first training sample from the
        re-shuffled training data.

        Actions needed to reset the environment:
        - Increment the episode counter by 1.
        - Shuffle the data.
        - Set current state as the first training sample from re-shuffled data.
        - Set step counter equal to 0.
        - Set reward collected equal to 0.

        :return: The new state after resetting the environment.
        """
        self.episode_counter += 1
        self.shuffle_data()
        self.current_state = self.x[0]
        self.step_counter = 0
        self.total_reward = 0
        return self.current_state

    def compute_reward(self, action: int, result: int):
        """
        :param action: Integer representing predicted casualty severity
        (0, 1, 2).
        :param result: Integer representing true casualty severity (0, 1, 2).

        :return: Reward based on prediction result and self.penalty_style
        instance attribute. Higher rewards are returned for correct predictions
        of lower frequency severity types and higher penalties respectively for
        wrong predictions.
        """
        if self.penalty_style == "standard":
            return rewards.reward_with_standard_penalties(
                action,
                result,
                self.rewards_
            )
        elif self.penalty_style == "adjusted":
            return rewards.reward_adjusted_penalties(
                action,
                result,
                self.rewards_)
        else:
            raise ValueError("Wrong value for penalty style")

    def is_done(self, reward, step_number: int):
        """
        Indicates whether the current episode is completed or not.

        :param reward: Reward returned from an action.
        :param step_number: Amount of steps taken so far in current episode.
        Each step is a prediction of casualty severity for one sample of the
        training data.

        :return: Returns a Boolean variable:
        True if the episode is completed or if a fatal / severe casualty
        has been misclassified.
        False if nothing of the above has happened.

        If the reward is smaller than -minimum reward * 2, it means that a
        sample with fatal or severe casualty severity has been misclassified.
        The minimum reward is returned for classification of light accidents.
        """
        min_reward = min(self.rewards_.values())
        terminal_limit = -2 * min_reward
        return any([
            reward < terminal_limit,
            self.samples - step_number == 1
        ])

    def step(self, action: int):
        """
        Takes a step within the Accidents environment and the current episode.

        :param action: Integer representing predicted casualty severity
        (0, 1, 2).

        :return: Tuple with the following elements:
        observation: The next observation within the training data, indexed by
        the step counter.
        reward: Reward returned for prediction of casualty severity for the
        current training sample (before step counter increment and return of
        the next observation).
        done: Boolean flag indicating whether the episode is completed or not.
        label: The true value of casualty severity.
        severity: Severity type ("light", "severe" or "fatal").
        """
        label = self.y[self.step_counter]
        severity = self.casualty_sev_dict[label]
        reward = self.compute_reward(action, label)
        self.total_reward += reward

        done = self.is_done(reward, self.step_counter)
        if done:
            self.render()
            observation = self.reset()
        else:
            self.step_counter += 1
            observation = self.x[self.step_counter]
        return observation, reward, done, (label, severity)

    def render(self, mode="human"):
        """
        :param mode: Parameter coming gym.Env default render method.

        Notify the user that current episode has been completed and print
        extra information:
        - Total number of casualty severity predictions made.
        - Total reward collected from predictions within the episode.
        """
        print(
            f"Episode {self.episode_counter} finished!",
            f"Total predictions made: {self.step_counter}",
            f"Total reward collected: {round(self.total_reward, 5)}",
            sep=" | "
        )
