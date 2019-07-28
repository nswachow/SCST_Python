import numpy as np
from scipy.stats import entropy
from functools import partial

from process_utils.pmf import PMF, JointPMF, ConditionalPMF
from test_tools import exceptionTest


def testPMF():

    LOW = 0
    HIGH = 10

    value_array = np.random.randint(low=LOW, high=HIGH, size=100)
    pmf = PMF(value_array)

    # Ensure probabilities sum to 1.0 and entropy is correctly calculated
    total_prob = 0.0
    entropy = 0.0
    for value in np.arange(LOW, HIGH):
        prob = pmf(value)
        total_prob += prob
        entropy -= prob * np.log2(prob)

    assert np.isclose(total_prob, 1.0)
    assert np.isclose(entropy, pmf.entropy)


def testJointPMF():

    LOW = 0
    HIGH = 10

    x_array = np.random.randint(low=LOW, high=HIGH, size=100)
    y_array = np.random.randint(low=LOW, high=HIGH, size=100)

    joint_pmf = JointPMF(x_array, y_array)
    x_pmf = PMF(x_array)
    y_pmf = PMF(y_array)

    tuple_array = [(x, y) for x, y in zip(x_array, y_array)]
    unique_tuples = list(set(tuple_array))

    # Ensure probabilities sum to 1.0 and mutual information is correctly calculated
    total_prob = 0.0
    mi = 0.0
    for value_tuple in unique_tuples:
        joint_prob = joint_pmf(value_tuple)
        x_prob = x_pmf(value_tuple[0])
        y_prob = y_pmf(value_tuple[1])

        total_prob += joint_prob
        mi += joint_prob * np.log2(joint_prob / (x_prob * y_prob))

    assert np.isclose(total_prob, 1.0)
    assert np.isclose(mi, joint_pmf.mutual_information)

    # Should fail to create JointPMF with invalid inputs
    exceptionTest(partial(JointPMF, x_array=np.arange(1), y_array=np.arange(2)),
                  AssertionError)


def testConditionalPMF():

    LOW = 0
    HIGH = 3
    NUM_DEPEND = 3

    value_array = np.random.randint(low=LOW, high=HIGH, size=100)
    depend_matrix = np.random.randint(low=LOW, high=HIGH, size=(NUM_DEPEND, 100))
    pmf = ConditionalPMF(value_array, depend_matrix)

    unique_depend = list(set([tuple(depend_matrix[:, ii])
                              for ii in range(depend_matrix.shape[1])]))

    # Ensure probabilities sum to 1.0 for each dependent variable tuple
    for depend_tuple in unique_depend:
        total_prob = 0.0
        for value in np.arange(LOW, HIGH):
            prob = pmf(value, depend_tuple)
            total_prob += prob

        assert np.isclose(total_prob, 1.0)


if __name__ == '__main__':
    testPMF()
    testJointPMF()
    testConditionalPMF()
