"""
Module with unit tests to check code responsible for computing rewards for the
reinforcement learning agent.
"""

import unittest
from reinforcement_learning import rewards

# dummy data
counts = {2: 100987, 0: 978, 1: 9089, 3: 1008989, 4: 478044}
test_rewards = {0: 90, 1: 10, 2: 1}
rng = range(len(test_rewards))


class TestRewards(unittest.TestCase):
    def test_prop_rev_rewards(self):
        """Test the function for computing rewards reverse to class ratio."""

        calc_rewards = rewards.prop_rev_rewards(counts)

        # Calculate rewards manually
        sum_ = sum(counts.values())
        manual_rewards = {
            0: counts[3] / sum_,
            1: counts[4] / sum_,
            2: counts[2] / sum_,
            3: counts[0] / sum_,
            4: counts[1] / sum_
        }

        for i in range(5):
            self.assertAlmostEqual(calc_rewards[i], manual_rewards[i], places=3)

    def test_rewards_from_minority(self):
        """Test the function for computing rewards based on minority class."""

        calc_rewards = rewards.rewards_from_minority(counts)

        # Calculate rewards manually
        manual_rewards = {
            0: counts[3] / counts[3],
            1: counts[4] / counts[3],
            2: counts[2] / counts[3],
            3: counts[0] / counts[3],
            4: counts[1] / counts[3],
        }

        for i in range(5):
            self.assertAlmostEqual(calc_rewards[i], manual_rewards[i], places=3)

    def test_standard_penalties(self):
        """Test reward_with_standard_penalties function."""
        calc_rewards = {}
        for i in rng:
            for j in range(i):
                key = f"{i}{j}"
                calc_rewards[key] = rewards.reward_with_standard_penalties(
                    i, j, test_rewards
                )

        # Calculate rewards manually
        manual_rewards = {
            "00": 90,
            "01": -10,
            "02": -1,
            "10": -90,
            "11": 10,
            "12": -1,
            "20": -90,
            "21": -10,
            "22": 1
        }

        for key in calc_rewards.keys():
            self.assertAlmostEqual(
                calc_rewards[key],
                manual_rewards[key],
                places=3
            )

    def test_adjusted_penalties(self):
        """Test reward_adjusted_penalties function."""

        calc_rewards = {}
        for i in rng:
            for j in range(i):
                key = f"{i}{j}"
                calc_rewards[key] = rewards.reward_adjusted_penalties(
                    i, j, test_rewards
                )

        # Calculate rewards manually
        manual_rewards = {
            "00": 90,
            "01": -10,
            "02": -1,
            "10": -80,
            "11": 10,
            "12": -1,
            "20": -89,
            "21": -9,
            "22": 1
        }

        for key in calc_rewards.keys():
            self.assertAlmostEqual(
                calc_rewards[key],
                manual_rewards[key],
                places=3
            )


if __name__ == '__main__':
    unittest.main()
