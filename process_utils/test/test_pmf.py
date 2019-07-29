import numpy as np
from scipy.stats import entropy
from functools import partial

from process_utils.pmf import ContiguousPMF, JointContiguousPMF, ConditionalContiguousPMF
from test_tools import exceptionTest


def testPMF():

    HIGH = 5
    MIN_PROB = 1e-4

    value_array = np.random.randint(low=0, high=HIGH, size=100)
    pmf = ContiguousPMF(value_array, MIN_PROB, HIGH)

    # Ensure probabilities sum to 1.0 and entropy is correctly calculated
    total_prob = 0.0
    entropy = 0.0
    for value in np.arange(pmf.max_value+1):
        prob = pmf(value)
        assert prob > 0
        total_prob += prob
        entropy -= prob * np.log2(prob)

    assert np.isclose(total_prob, 1.0)
    assert np.isclose(entropy, pmf.entropy)

    exceptionTest(lambda: pmf(-1), KeyError)

    # Should fail to create ContiguousPMF with invalid inputs
    exceptionTest(partial(ContiguousPMF, train_array=np.arange(1), min_prob=-1, max_value=HIGH),
                  AssertionError)
    exceptionTest(partial(ContiguousPMF, train_array=np.arange(1), min_prob=1, max_value=-1),
                  AssertionError)
    exceptionTest(partial(ContiguousPMF, train_array=np.arange(5), min_prob=1e-4, max_value=1),
                  AssertionError)


def testJointPMF():

    HIGH = 5

    x_array = np.random.randint(low=0, high=HIGH, size=100)
    y_array = np.random.randint(low=0, high=HIGH, size=100)

    joint_pmf = JointContiguousPMF(x_array, y_array, HIGH)
    x_pmf = ContiguousPMF(x_array, 0, HIGH)
    y_pmf = ContiguousPMF(y_array, 0, HIGH)
    tuple_array = [(x, y) for x in np.arange(HIGH+1) for y in np.arange(HIGH+1)]

    # Ensure probabilities sum to 1.0 and properties are correctly calculated
    total_prob = 0.0
    entropy = 0.0
    mi = 0.0
    for value_tuple in tuple_array:
        joint_prob = joint_pmf(value_tuple)
        assert joint_prob >= 0
        x_prob = x_pmf(value_tuple[0])
        y_prob = y_pmf(value_tuple[1])

        total_prob += joint_prob

        if joint_prob > 0:
            entropy -= joint_prob * np.log2(joint_prob)
            if x_prob > 0 and y_prob > 0:
                mi += joint_prob * np.log2(joint_prob / (x_prob * y_prob))

    assert np.isclose(total_prob, 1.0)
    assert np.isclose(entropy, joint_pmf.entropy)
    assert np.isclose(mi, joint_pmf.mutual_information)

    # Should fail to create JointContiguousPMF with invalid inputs
    exceptionTest(partial(JointContiguousPMF, x_train_array=np.arange(1),
                          y_train_array=np.arange(2), max_value=HIGH), AssertionError)


def testConditionalPMF():

    HIGH = 3
    NUM_DEPEND = 3
    MIN_PROB = 1e-4

    value_array = np.random.randint(low=0, high=HIGH, size=100)
    depend_matrix = np.random.randint(low=0, high=HIGH, size=(NUM_DEPEND, 100))
    pmf = ConditionalContiguousPMF(value_array, depend_matrix, MIN_PROB, HIGH)

    unique_depend = list(set([tuple(depend_matrix[:, ii])
                              for ii in range(depend_matrix.shape[1])]))

    # Ensure probabilities sum to 1.0 for each dependent variable tuple
    for depend_tuple in unique_depend:
        total_prob = 0.0
        for value in np.arange(pmf.max_value+1):
            prob = pmf(value, depend_tuple)
            total_prob += prob

        assert np.isclose(total_prob, 1.0)

    # Test with untrained and invalid dependencies
    assert np.isclose(pmf(0, (HIGH, HIGH, HIGH)),  1 / HIGH)
    exceptionTest(lambda: pmf(0, (-1, 0, 0)), AssertionError)
    exceptionTest(lambda: pmf(0, (HIGH+1, 0, 0)), AssertionError)
    exceptionTest(lambda: pmf(0, (0,)), AssertionError)


if __name__ == '__main__':
    testPMF()
    testJointPMF()
    testConditionalPMF()
