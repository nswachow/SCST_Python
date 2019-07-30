import numpy as np
from typing import List, Tuple, Optional
from sys import float_info

from scst.pmf import JointContiguousPMF, ConditionalContiguousPMF


class SequenceIndex:
    '''
    Indexes a past vector (relative to the current one) and an element in that vector.
    '''

    def __init__(self, vector: int, element: int):
        self._vector = vector
        self._element = element

    @property
    def vector(self) -> int:
        '''
        Vector index back in time relative to some base index.
        '''
        return self._vector

    @property
    def element(self) -> int:
        '''
        Index for the element of the vector associated with self._vector.
        '''
        return self._element

    def index(self, base_idx: int) -> Tuple[int, int]:
        '''
        Get a tuple that can be used for indexing a matrix of column vectors, relative to an input
        vector index.
        '''
        return (self._element, base_idx-self._vector)


class SCSTElementModel:
    '''
    For for a single vector element in the SCST classifier. Can be used to calculate conditional
    log-likelihoods for a value given a set of dependent values, which in turn can be used to
    calculate the conditional log-likelihood of an entire vector.
    '''

    def __init__(self, train_event_list: List[np.ndarray], element_idx: int, order: int,
                 max_num_depend: int, mi_thresh: float, min_prob: float, max_value: int,
                 num_pri_obs: Optional[int]):
        '''
        :param train_event_list: Element are matrices representing (quantized) vector sequences
            (columns of each matrix) associated with a specific class label.
        :param element_idx: Index that this element represents within a observation vector.
        :param order: Maximum number of previous vectors in a sequence this model can depend on.
        :param max_num_depend: Maximum number of elements the this model can depend on.
        :param mi_thresh: Threshold on mutual information used to determine if the element this
            model represents depends on another element.
        :param min_prob: Minimum probability output by this model. Used so probabilities of zero
            aren't output when novel data is encountered.
        :param max_value: Maximum value that can be observed for any element in any event in
            ``train_event_list``.
        :param num_pri_obs: Number of observations at the beginning of each event to use for finding
            prior probabilities. The entirety of each event is used if this value is None.
        '''
        assert len(train_event_list) > 0
        assert (element_idx >=0) and (element_idx < train_event_list[0].shape[0])
        assert order >= 0
        assert max_num_depend >= 0
        assert mi_thresh >= 0
        assert num_pri_obs is None or num_pri_obs >= 0

        self._element_idx = element_idx
        self._vect_dim = train_event_list[0].shape[0]
        self._order = order

        train_event_list = SCSTElementModel._getPriorEventList(train_event_list, num_pri_obs)
        self._depend_idx = self._learnDependentIndicies(
            train_event_list, max_num_depend, mi_thresh, max_value)
        self._pmf = self._learnConditionalPMF(train_event_list, min_prob, max_value)

    def logLikelihood(self, data_matrix: np.ndarray) -> float:
        '''
        Generate the log-likelihood of this model given a vector sequence.

        :param data_matrix: Columns are feature vectors ordered by time, with the first column
            being the "oldest" and the last column being the "current" vector.

        :return: The conditional log-likelihood of this model given the sequence ``data_matrix``.
        '''

        assert data_matrix.shape == (self._vect_dim, self._order + 1)

        depend_list = []
        for seq_idx in self._depend_idx:
            depend_list.append(data_matrix[seq_idx.index(-1)])

        depend_tuple = tuple(depend_list)

        likelihood = self._pmf(data_matrix[self._element_idx, -1], depend_tuple)

        if likelihood < float_info.min:
            # Avoid numerical issue in case min_prob == 0
            return float_info.min
        else:
            return np.log(likelihood)

    def _learnConditionalPMF(self, train_event_list: List[np.ndarray], min_prob: float,
                             max_value: int) -> ConditionalContiguousPMF:
        '''
        Get a conditional mass function to represent this element.
        '''

        train_array = self._getElementSamples(train_event_list, SequenceIndex(0, self._element_idx))
        num_samples = len(train_array)
        depend_matrix = np.array([]).reshape((0, num_samples))

        for seq_idx in self._depend_idx:
            depend_row = self._getElementSamples(train_event_list, seq_idx)
            depend_matrix = np.concatenate((depend_matrix, depend_row.reshape((1, num_samples))))

        return ConditionalContiguousPMF(train_array, depend_matrix, min_prob, max_value)

    def _learnDependentIndicies(self, train_event_list: List[np.ndarray], max_num_depend: int,
                                mi_thresh: float, max_value: int) -> List[SequenceIndex]:
        '''
        Get a List of SequenceIndex that represent that relative elements in a vector sequence that
        this element is found to be most dependent on.
        '''

        this_samples = self._getElementSamples(train_event_list,
                                               SequenceIndex(0, self._element_idx))

        # Generate a SequenceIndex for each vector element in the current and previous vectors that
        # this element is found to be dependent on.
        mi_list = []
        mi_idx_list = []
        for vector_idx in range(self._order + 1):
            max_depend_idx = self._element_idx if vector_idx == 0 else self._vect_dim
            for depend_idx in range(max_depend_idx):

                other_idx = SequenceIndex(vector_idx, depend_idx)
                other_samples = self._getElementSamples(train_event_list, other_idx)
                joint_pmf = JointContiguousPMF(this_samples, other_samples, max_value)
                if joint_pmf.mutual_information > mi_thresh:
                    mi_list.append(joint_pmf.mutual_information)
                    mi_idx_list.append(other_idx)

        # Select the top max_num_depend elements to be dependent on this one
        if len(mi_list) > max_num_depend:
            hi_mi_idx = np.argsort(mi_list)[-max_num_depend:]
            mi_idx_list = [mi_idx_list[idx] for idx in hi_mi_idx]

        return mi_idx_list

    def _getElementSamples(self, train_event_list: List[np.ndarray],
                           seq_idx: SequenceIndex) -> np.ndarray:
        '''
        Get an array of all the samples in all events in ``train_event_list`` associated with a the
        intput ``seq_idx``.
        '''

        all_samples = np.array([])
        for event in train_event_list:
            if event.shape[1] < self._order:
                print("Warning: Event length is shorter than model order. Ignoring event.")
                continue

            end_idx = -seq_idx.vector if seq_idx.vector > 0 else None
            sample_vect = event[seq_idx.element, self._order-seq_idx.vector:end_idx]
            all_samples = np.append(all_samples, sample_vect)

        return all_samples

    @staticmethod
    def _getPriorEventList(train_event_list: List[np.ndarray],
                           num_pri_obs: Optional[int]) -> List[np.ndarray]:
        '''
        Get a list of events based on the input ``train_event_list``, but where each event contains
        at most the ``num_pri_obs`` vectors at the beginning of the event.
        '''

        if num_pri_obs is None:
            return train_event_list

        new_event_list = []
        for event in train_event_list:
            end_idx = min(num_pri_obs, event.shape[1])
            new_event_list.append(event[:, :end_idx])

        return new_event_list
