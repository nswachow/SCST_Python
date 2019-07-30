import numpy as np
import scipy.stats
from collections import defaultdict
from typing import Tuple, Dict, Iterable


class ContiguousPMF:
    '''
    Implements a probability mass function for a discrete random variable whose realizations are
    integers on the interval [0, ``max_value``] where ``max_value`` is an input. The user also
    specifies a minimum probability for all integers in the interval, such that the probability
    will be at least this large for all values, even those that don't appear in the array of
    training data.
    '''

    def __init__(self, train_array: np.ndarray, min_prob: float, max_value: int):
        '''
        :param train_array: Array of realizations that this PMF represents, from which the mass
            function shall be computed.
        :param min_prob: The minimum probabilty assigned to each value \in [0, max_value].
        :param max_value: The maximum value associated with this PMF, which needs to be specified
            since train_array may not contain any realizations of this values, or others on the
            relevant interval.
        '''

        assert min_prob >= 0
        assert min_prob < 1 / max_value, "Cannot satisfy min_prob criterion"
        assert max_value >= 0

        train_array = train_array.astype(int)
        assert np.all(train_array <= max_value)
        assert np.all(train_array >= 0)

        value_array = np.arange(max_value + 1)
        self._prob_dict = ContiguousPMF._getProbabilityDict(train_array, value_array, min_prob)
        self._max_value = max_value

        self._entropy = None  # Compute lazily

    def __call__(self, value: int) -> float:
        '''
        Evaluate the PMF at the input value.

        :param value: Realization value.

        :return: Probability of the input value.
        '''
        # We purposely want to throw a key error when using an invalid or out of range key
        return self._prob_dict[value]

    @property
    def max_value(self) -> int:
        '''
        The maximum value associated with this PMF, such that this PMF only has non-zero mass for
        values >= 0 and <= max_value.
        '''
        return self._max_value

    @property
    def entropy(self) -> float:
        '''
        :return: The entropy of the random variable represented by this PMF.
        '''

        if self._entropy is None:
            non_zero_values = np.array(list(self._prob_dict.values()))
            non_zero_values = non_zero_values[np.nonzero(non_zero_values)[0]]
            self._entropy = scipy.stats.entropy(non_zero_values, base=2)

        return self._entropy

    @staticmethod
    def _getProbabilityDict(train_array: np.ndarray, value_array: np.ndarray,
                            min_prob: float) -> Dict[int, float]:
        '''
        Helper function to allow for generating probabilites for this class and derived classes
        '''

        prob_dict = defaultdict(lambda: 0.0)  # zero probability where there is no mass
        prob_per_value = 1 / len(train_array)
        for value in train_array:
            prob_dict[value] += prob_per_value

        ContiguousPMF._enforceMinProb(prob_dict, value_array, min_prob)

        # Convert to a normal dict so we don't add every value we try from this point forward
        return dict(prob_dict)

    @staticmethod
    def _enforceMinProb(prob_dict: defaultdict, value_array: np.ndarray,
                        min_prob: float) -> None:
        '''
        Enforce minimum probability constraint by setting probabilities < min_prob to min_prob
        and rescaling the remaining probabilties so the mass sums to one.
        '''

        values_to_normalize = []
        num_small_prob = 0
        norm_mass = 0.0

        # Run through this process even if min_prob == 0 so that probabilities are generated for all
        # the possible values.
        for value in value_array:
            prob = prob_dict[value]
            if prob < min_prob:
                prob_dict[value] = min_prob
                num_small_prob += 1
            else:
                values_to_normalize.append(value)
                norm_mass += prob

        if num_small_prob > 0:
            prob_scale = (1 - num_small_prob * min_prob) / norm_mass
            for value in values_to_normalize:
                prob_dict[value] *= prob_scale


class JointContiguousPMF(ContiguousPMF):
    '''
    Implements a joint probability mass function that functions similarly to ContiguousPMF, but
    represents the joint distribution of two random variables. Can be used to obtain probabilities
    when called with a pair of values. Minimum probabilities aren't enforced in this case, since it
    causes discrepencies between the marginal and joint distributions.
    '''

    def __init__(self, x_train_array: np.ndarray, y_train_array: np.ndarray, max_value: int):
        '''
        :param x_train_array, y_train_array: Arrays of realizations for the first and second
            dimension that this joint PMF represents, from which the mass function shall be
            computed. Elements of each array have one-to-one correspondance, i.e., element ``ii``
            of each array are assumed to have been realized simultaneously. These arrays must have
            the same length.
        :param max_value: See ContiguousPMF.__init__.
        '''

        assert len(x_train_array) == len(y_train_array)
        self._x_marginal = ContiguousPMF(x_train_array, 0, max_value)
        self._y_marginal = ContiguousPMF(y_train_array, 0, max_value)

        train_array = [(x, y) for x, y in zip(x_train_array.astype(int), y_train_array.astype(int))]
        value_array = [(x, y) for x in np.arange(max_value + 1) for y in np.arange(max_value + 1)]
        self._prob_dict = ContiguousPMF._getProbabilityDict(train_array, value_array, 0)
        self._max_value = max_value

        self._entropy = None  # Compute lazily
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


class ConditionalContiguousPMF:
    '''
    Implements a conditional probability mass function that can be used to obtain probabilities when
    called with a tuple of dependent values (to determine the PMF to draw from) and a value at which
    to evaluate the probability.
    '''

    def __init__(self, train_array: np.ndarray, depend_matrix: np.ndarray, min_prob: float,
                 max_value: int):
        '''
        :param train_array: Array of realizations that this PMF represents, from which the mass
            function shall be computed.
        :param depend_matrix: {ii}th column represents the values of the dependent variables
            when the {ii} element of ``train_array`` was realized. This is a matrix to enforce the
            same number of dependent values per realization.
        :param min_prob, max_value: See ContiguousPMF.__init__.
        '''

        assert len(train_array) == depend_matrix.shape[1]

        # Construct a value array for each dependent value
        value_dict = defaultdict(list)
        for ii in range(len(train_array)):
            value_dict[tuple(depend_matrix[:, ii])].append(train_array[ii])

        # Construct a PMF for each dependent value
        self._pmf_dict = {}
        for depend_values, pmf_value_list in value_dict.items():
            self._pmf_dict[depend_values] = ContiguousPMF(np.array(pmf_value_list), min_prob,
                                                          max_value)

        self._max_value = max_value
        self._num_depend = depend_matrix.shape[0]
        self._uniform_prob = 1 / max_value

    @property
    def max_value(self) -> int:
        '''
        The maximum value associated with this PMF, such that this PMF only has non-zero mass for
        values >= 0 and <= max_value.
        '''
        return self._max_value

    @property
    def num_depend(self) -> int:
        '''
        The number of dependent values needed to calculate a conditional probability.
        '''
        return self._num_depend

    def __call__(self, value: float, depend_values: Tuple[Iterable[float]]) -> float:
        '''
        Evaluate the conditional PMF according to the inputs. Return a uniform probability on the
        interval [0, max_value] if there is no PMF associated with the input ``depend_values``.

        :param value: Realization value.
        :param depend_values: Values the PMF is conditioned on.

        :return: Conditional probability of the ``value`` input given the ``depend_values``.
        '''

        # Enforce structure on depend_values
        assert len(depend_values) == self._num_depend
        depend_array = np.array(depend_values)
        assert np.all(depend_array >= 0) and np.all(depend_array <= self._max_value)

        try:
            return self._pmf_dict[depend_values](value)
        except KeyError:
            return self._uniform_prob
