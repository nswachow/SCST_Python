import numpy as np
from collections import defaultdict
from scipy.stats import entropy
from typing import Tuple


class PMF:
    '''
    Implements a probability mass function that can be used to obtain probabilities when called with
    a value.
    '''

    def __init__(self, value_array: np.ndarray):
        '''
        :param value_array: Array of realizations that this PMF represents, from which the mass
            function shall be computed.
        '''

        prob_per_value = 1 / len(value_array)
        self._prob_dict = defaultdict(lambda: 0.0)  # zero probability where there is no mass

        for value in value_array:
            self._prob_dict[value] += prob_per_value

        # Convert to a normal dict so we don't add every value we try from this point forward
        self._prob_dict = dict(self._prob_dict)
        self._entropy = None  # Compute lazily

    def __call__(self, value: float) -> float:
        '''
        Evaluate the PMF at the input value.

        :param value: Realization value.

        :return: Probability of the input value.
        '''
        try:
            return self._prob_dict[value]
        except KeyError:
            return 0.0

    @property
    def entropy(self) -> float:
        '''
        :return: The entropy of the random variable represented by this PMF.
        '''

        if self._entropy is None:
            self._entropy = entropy(list(self._prob_dict.values()), base=2)

        return self._entropy


class JointPMF(PMF):
    '''
    Implements a joint probability mass function that can be used to obtain probabilities when
    called with a pair of values.
    '''

    def __init__(self, x_array: np.ndarray, y_array: np.ndarray):
        '''
        :param x_array, y_array: Arrays of realizations for the first and second dimension that
            this joint PMF represents, from which the mass function shall be computed. Elements of
            each array have one-to-one correspondance, i.e., element ``ii`` of each array are
            assumed to have been realized simultaneously. These arrays must have the same length.
        '''

        assert len(x_array) == len(y_array)

        tuple_array = [(x, y) for x, y in zip(x_array, y_array)]
        super().__init__(tuple_array)

        self._x_marginal = PMF(x_array)
        self._y_marginal = PMF(y_array)

        self._mutual_information = None  # Compute lazily

    @property
    def mutual_information(self) -> float:
        '''
        The mutual information of the two random variables associated with this joint PMF.
        '''

        if self._mutual_information is None:
            self._mutual_information = (self._x_marginal.entropy + self._y_marginal.entropy -
                                        self.entropy)

        return self._mutual_information


class ConditionalPMF:
    '''
    Implements a conditional probability mass function that can be used to obtain probabilities when
    called with a tuple of dependent values (to determine the PMF to draw from) and a value at which
    to evaluate the probability.
    '''

    def __init__(self, value_array: np.ndarray, depend_matrix: np.ndarray):
        '''
        :param value_array: Array of realizations that this PMF represents, from which the mass
            function shall be computed.
        :param depend_matrix: {ii}th column represents the values of the dependent variables
            when the {ii} element of ``value_array`` was realized. This is a matrix to enforce the
            same number of dependent values per realization.
        '''

        assert len(value_array) == depend_matrix.shape[1]

        # Construct a value array for each dependent value
        value_dict = defaultdict(list)
        for ii in range(len(value_array)):
            value_dict[tuple(depend_matrix[:, ii])].append(value_array[ii])

        # Construct a PMF for each dependent value
        self._pmf_dict = {}
        for depend_values, pmf_value_list in value_dict.items():
            self._pmf_dict[depend_values] = PMF(np.array(pmf_value_list))

    def __call__(self, value: float, depend_values: Tuple[float]) -> float:
        '''
        Evaluate the conditional PMF according to the inputs.

        :param value: Realization value.
        :param depend_values: Values the PMF is conditioned on.

        :return: Conditional probability of the ``value`` input given the ``depend_values``.
        '''
        try:
            return self._pmf_dict[depend_values](value)
        except KeyError:
            return 0.0
