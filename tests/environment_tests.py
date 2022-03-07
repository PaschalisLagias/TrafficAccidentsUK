"""
Module with unit tests to check code responsible for computing rewards for the
reinforcement learning agent.
"""

import unittest
import numpy as np

from reinforcement_learning.environment import Accidents

dummy_x = np.array([
    [1, 3, 4],
    [1, 3, 4],
    [1, 3, 4],
    [1, 3, 4],
    [1, 3, 4],
    [1, 3, 4],
    [1, 3, 4],
    [1, 3, 4],
    [1, 3, 4],
])

dummy_y = np.array([0, 1, 2, 2, 2, 2, 1, 2, 1])

dummy_env = Accidents(dummy_x, dummy_y)


class TestEnvironment(unittest.TestCase):
    def test_action_space(self):
        action = dummy_env.action_space.sample()
        self.assertIn(action, (0, 1, 2))

    def test_observation_space(self):
        obs = dummy_env.observation_space.sample()
        self.assertEqual(obs.shape[0], 1, "Observation unit check")
        self.assertEqual(obs.shape[1], 3, "Observation field check")

    def test_reset(self):
        pass

    def test_step(self):
        pass

    def test_is_done(self):
        pass


if __name__ == '__main__':
    unittest.main()

# TODO: Write tests and continue DRL-Hands On book at page 154.
