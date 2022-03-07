"""
File with reward functions for reinforcement learning agent.

Functions are based on the result of predicting the "Casualty_Severity"
attribute for an individual who is involved in a traffic accident.

Possible values and respective meaning for "Casualty_Severity":
- 0: Fatal
- 1: Severe
- 2: Light

The main concept is to give the RL agent higher rewards for predicting correctly
fatal and severe casualties, since we are more interested in these incidents,
and they represent a weak minority of total traffic accidents.
"""


def reverse_order(counts: dict):
    """
    :param counts: Dictionary - instance of collections.Counter class. Contains
    counts of fatal, severe and light accidents.

    :return: Tuple with 2 list:
    keys: List of counts dictionary keys, sorted by their respective values in
    ascending order.
    values: List of counts dictionary values, sorted in descending order.
    """
    items = sorted(counts.items(), key=lambda count: count[1], reverse=True)
    keys, values = zip(*items)
    keys = list(reversed(keys))
    return keys, values


def rewards_from_minority(counts: dict):
    """

    :param counts: Dictionary - instance of collections.Counter class. Contains
    counts of fatal, severe and light accidents.

    :return: Dictionary with class values as keys and rewards as values e.g.
    {0: 0.856, 1: 0.134, 2: 0.010}
    """
    max_frequency = max(counts.values())
    keys, values = reverse_order(counts)
    rng = range(len(keys))
    rewards = {keys[i]: values[i] / max_frequency for i in rng}
    return rewards


def prop_rev_rewards(counts: dict):
    """
    Get a list of tuples (counts key, counts value) sorted in descending order
    by counts value (the count of samples per severity type).
    Get the keys (severity types) from the above list, save them in a new list
    and reverse the order (ascending order by counts value).
    Get the counts values as ordered in the list of tuples and save them in a
    new list.

    Create rewards dictionary: Looping through the index of counts, divide the
    indexed count from values list by the total count of accidents, and assign
    it to the indexed key from keys list.

    This way, prediction of the lowest frequency severity type returns the
    highest reward, the second-lowest frequency the second-best reward etc.

    :param counts: Dictionary - instance of collections.Counter class. Contains
    counts of fatal, severe and light accidents.

    :return: Dictionary with class values as keys and rewards as values e.g.
    {0: 0.856, 1: 0.134, 2: 0.010}
    """
    counts_sum = sum(counts.values())
    keys, values = reverse_order(counts)
    rng = range(len(keys))
    rewards = {keys[i]: values[i] / counts_sum for i in rng}
    return rewards


def reward_with_standard_penalties(guess: int, real: int, rewards: dict):
    """
    :param guess: Casualty severity guess.
    :param real: Actual casualty severity from labelled data.
    :param rewards: Dictionary with severity types (represented by integers) as
    keys, and the respective rewards as values.

    :return: Reward or penalty based on severity guess outcome. Higher reward
    is returned for correct prediction of lower frequency class. Penalties are
    returned for incorrect prediction, equal to the reward of the real severity
    type, multiplied by -1.
    """
    if guess == real:
        return rewards[real]
    return -rewards[real]


def reward_adjusted_penalties(guess: int, real: int, rewards: dict):
    """
    :param guess: Casualty severity guess.
    :param real: Actual casualty severity from labelled data.
    :param rewards: Dictionary with severity types (represented by integers) as
    keys, and the respective rewards as values.

    :return: Reward or penalty based on severity guess outcome. Higher reward
    is returned for correct prediction of lower frequency class. Penalties are
    returned for incorrect prediction:
    If predicted class returns higher reward than the real one,
    penalty = real penalty * -1.
    If real class returns higher reward than the predicted one,
    penalty = (real reward - predicted reward) * -1.
    """
    if guess == real:
        return rewards[real]
    elif rewards[guess] > rewards[real]:
        return -rewards[real]
    elif rewards[guess] < rewards[real]:
        return -(rewards[real] - rewards[guess])
