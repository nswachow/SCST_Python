import numpy as np
from typing import Callable, Tuple, Dict, Any, List


DEFAULT_A_LOW = -3
DEFAULT_A_HIGH = 0.5
DEFAULT_B_LOW = -0.5
DEFAULT_B_HIGH = 0.5
DEFAULT_NOISE_LOW = -0.5
DEFAULT_NOISE_HIGH = 3


def getTrainDict(obs_per_event: int, events_per_class: int, input_dim: int,
                 a_low: float=DEFAULT_A_LOW, a_high: float=DEFAULT_A_HIGH,
                 b_low: float=DEFAULT_B_LOW, b_high: float=DEFAULT_B_HIGH,
                 noise_low: float=DEFAULT_NOISE_LOW,
                 noise_high: float=DEFAULT_NOISE_HIGH) -> Dict[Any, List[np.ndarray]]:
    '''
    Generate and return a dictionary of training event data that can be used to train, e.g., a
    ``SCSTClassifier``.

    :param obs_per_event: Length of each training event (identical for all events).
    :param events_per_class: The number of training events per class.
    :param input_dim: The dimensionality of the vectors in each sequence.
    :param a_low, a_high: Lower and upper bounds on the values of the 'A' class, respectively.
    :param b_low, b_high: Lower and upper bounds on the values of the 'B' class, respectively.
    :param noise_low, noise_high: Lower and upper bounds on the values of the noise class (with
        label "0"), respectively.

    :return: A dictionary of lists, keyed by class label, where lists containing training events
        (matrices) for the corresponding class.
    '''

    train_event_dict = {}
    event_size = (input_dim, obs_per_event)

    train_event_dict['A'] = [np.random.uniform(low=a_low, high=a_high, size=event_size)
                             for _ in range(events_per_class)]
    train_event_dict['B'] = [np.random.uniform(low=b_low, high=b_high, size=event_size)
                             for _ in range(events_per_class)]
    train_event_dict[0] = [np.random.uniform(low=noise_low, high=noise_high, size=event_size)
                           for _ in range(events_per_class)]

    return train_event_dict


def getTestSequence(input_dim: int, a_low: float=DEFAULT_A_LOW, a_high: float=DEFAULT_A_HIGH,
                    b_low: float=DEFAULT_B_LOW, b_high: float=DEFAULT_B_HIGH,
                    noise_low: float=DEFAULT_NOISE_LOW,
                    noise_high: float=DEFAULT_NOISE_HIGH) -> Tuple[np.array, np.array]:
    '''
    Generate and return data that can be used to test a transient classifier, e.g., a
    ``SCSTClassifier``.

    :param input_dim: The dimensionality of the vectors in each sequence.
    :param a_low, a_high: Lower and upper bounds on the values of the 'A' class, respectively.
    :param b_low, b_high: Lower and upper bounds on the values of the 'B' class, respectively.
    :param noise_low, noise_high: Lower and upper bounds on the values of the noise class (with
        label "0"), respectively.

    :return: The first element is a matrix representing a test vector sequence, where feature
        vectors are columns of this matrix. The second element is an an array representing the true
        class labels of the vectors in the test sequence.
    '''

    length_list = [20, 50, 40, 70, 30]

    test_array = np.concatenate((
        np.random.uniform(low=noise_low, high=noise_high, size=((input_dim, length_list[0]))),
        np.random.uniform(low=a_low, high=a_high, size=((input_dim, length_list[1]))),
        np.random.uniform(low=noise_low, high=noise_high, size=((input_dim, length_list[2]))),
        np.random.uniform(low=b_low, high=b_high, size=((input_dim, length_list[3]))),
        np.random.uniform(low=noise_low, high=noise_high, size=((input_dim, length_list[4])))),
        axis=1)

    test_labels = np.concatenate((
        np.array([0] * length_list[0]),
        np.array(['A'] * length_list[1]),
        np.array([0] * length_list[2]),
        np.array(['B'] * length_list[3]),
        np.array([0] * length_list[4])
    ))

    return test_array, test_labels


def exceptionTest(test_func: Callable, except_type: Exception):
    '''
    Used in unit tests to ensure a given function appropriately throws an acception of a specific
    type.
    '''

    was_thrown = False
    try:
        test_func()
    except except_type:
        was_thrown = True

    assert was_thrown
