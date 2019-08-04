import abc
from typing import Tuple, Union
import numpy as np
from sklearn import cluster


class Quantizer(metaclass=abc.ABCMeta):
    '''
    Abstract base class with a __call__ method that takes a value as an input and outputs a value
    that is a member of a predetermined set, and is closest to the input according to some measure.
    '''

    def __init__(self, num_levels: int, value_array: np.ndarray, is_discrete: bool):
        '''
        :param num_levels: The number of quantization levels.
        :param value_array: Used to train the quantizer (calculate transition and reconstruction
            levels).
        :param is_discrete: If True, then reconstruction levels will be \in [0, num_levels-1],
            rather assuming float values.
        '''

        assert num_levels >= 2
        self._num_levels = num_levels
        self._is_discrete = is_discrete
        self._transition_levels, self._reconstruction_levels = self._getLevels(value_array)

        # Since we're leaving it up to derived classes to calculate levels, verify their lengths
        # here
        assert len(self._reconstruction_levels) == self._num_levels
        assert len(self._transition_levels) == self._num_levels - 1

    @property
    def num_levels(self) -> int:
        '''
        The number of quantization levels.
        '''
        return self._num_levels

    @property
    def transition_levels(self) -> np.ndarray:
        '''
        Used to determine what bin a value falls into, i.e., which element of
        ``reconstruction_levels`` should represent a given input value. Returns a copy so the
        internal array can't be modified.
        '''
        return self._transition_levels.copy()

    @property
    def reconstruction_levels(self) -> np.ndarray:
        '''
        Values that inputs to the __call__ method are quantized to. Returns a copy so the internal
        array can't be modified.
        '''
        return self._reconstruction_levels.copy()

    def __call__(self, values: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        '''
        Perform quantization.

        :param values: Value or array of values to quantize.

        :return: Quantized result.
        '''

        level_idx = np.digitize(values, self._transition_levels)

        if self._is_discrete:
            return level_idx
        else:
            return self._reconstruction_levels[level_idx]

    @abc.abstractmethod
    def _getLevels(self, value_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Abstract method for generating and returning a tuple of the transition and reconstruction
        levels for this quantizer, in that order.

        :param value_array: see self.__init__.
        '''
        pass


class LloydMaxQuantizer(Quantizer):
    '''
    Implements the Lloyd-Max quantizer, i.e., a quantizer that yields the minimum mean squared error
    between input and quantized values, assuming the input values are drawn from the same
    distribution as the elements of ``value_array`` used to train this quantizer.
    '''

    def __init__(self, num_levels: int, value_array: np.ndarray, is_discrete: bool,
                 num_initializations: int=10):
        '''
        :param num_levels, value_array, is_discrete: See Quantizer.__init__
        :param num_initializations: Number of iterations to use in attempt to find the optimal
            quantizer.
        '''
        assert num_initializations >= 1
        self._num_initializations = num_initializations
        super().__init__(num_levels, value_array, is_discrete)

    def _getLevels(self, value_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' override '''

        # Calculate reconstruction levels
        clusterer = cluster.KMeans(n_clusters=self._num_levels, n_init=self._num_initializations)
        clusterer.fit(value_array.reshape(len(value_array), 1))
        reconstruction_levels = np.sort(clusterer.cluster_centers_.flatten())

        # Calculate transition levels as the values mid-way between reconstruction levels
        transition_levels = (reconstruction_levels[1:] + reconstruction_levels[:-1]) / 2

        return transition_levels, reconstruction_levels
