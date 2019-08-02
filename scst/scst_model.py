import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Type
from sys import float_info
from collections import defaultdict
import copy

from scst.pmf import JointContiguousPMF, ConditionalContiguousPMF
from scst.quantizer import Quantizer


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
    Models a single vector element in the SCST classifier. Can be used to calculate conditional
    log-likelihoods for a value given a set of dependent values, which in turn can be used to
    calculate the conditional log-likelihood of an entire vector.
    '''

    def __init__(self, train_event_list: List[np.ndarray], element_idx: int, order: int,
                 max_num_depend: int, mi_thresh: float, min_prob: float, max_value: int):
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
        '''

        self._element_idx = element_idx
        self._vect_dim = train_event_list[0].shape[0]
        self._order = order

        assert len(train_event_list) > 0
        assert (element_idx >= 0) and (element_idx < self._vect_dim)
        assert order >= 0
        assert max_num_depend >= 0
        assert mi_thresh >= 0

        self._depend_idx = self._learnDependentIndicies(
            train_event_list, max_num_depend, mi_thresh, max_value)
        self._pmf = self._learnConditionalPMF(train_event_list, min_prob, max_value)

    def logLikelihood(self, data_matrix: np.ndarray) -> float:
        '''
        Generate the log-likelihood of this element model given a vector sequence.

        :param data_matrix: Columns are feature vectors ordered by time, with the first column
            being the "oldest" and the last column being the "current" vector, an element of which
            we shall evaluate the likelihood of this model with respect to.

        :return: The conditional log-likelihood of a given element in the last column vector in
            ``data_matrix`` given this model and previous column vectors ``data_matrix``.
        '''

        assert data_matrix.shape == (self._vect_dim, self._order + 1)

        depend_list = []
        for seq_idx in self._depend_idx:
            depend_list.append(data_matrix[seq_idx.index(-1)])

        depend_tuple = tuple(depend_list)

        likelihood = self._pmf(data_matrix[self._element_idx, -1], depend_tuple)

        if likelihood < float_info.min:
            # Avoid warning associated with computing log(0)
            return -np.inf
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


class SCSTVectorModel:
    '''
    Models an entire vector in the SCST classifier. Can be used to calculate conditional
    log-likelihoods for a vector given previous vectors in the sequence.
    '''

    def __init__(self, train_event_list: List[np.ndarray], order: int, max_num_depend: int,
                 mi_thresh: float, min_prob: float, max_value: int):
        '''
        :param train_event_list, order, max_num_depend, mi_thresh, min_prob, max_value:
            See SCSTElementModel.__init__.
        '''

        assert len(train_event_list) > 0
        assert order >= 0

        self._vect_dim = train_event_list[0].shape[0]
        self._order = order

        self._model_list = []
        for element_idx in range(self._vect_dim):
            element_model = SCSTElementModel(train_event_list, element_idx, order, max_num_depend,
                                             mi_thresh, min_prob, max_value)
            self._model_list.append(element_model)

    def logLikelihood(self, data_matrix: np.ndarray) -> float:
        '''
        Generate the log-likelihood of this vector model given a vector sequence.

        :param data_matrix: Columns are feature vectors ordered by time, with the first column
            being the "oldest" and the last column being the "current" vector that we shall
            evaluate the likelihood of this model with respect to.

        :return: The conditional log-likelihood of the last column vector in ``data_matrix`` given
            this model and previous column vectors ``data_matrix``.
        '''

        assert data_matrix.shape == (self._vect_dim, self._order + 1)

        log_likelihood = 0.0
        for model in self._model_list:
            log_likelihood += model.logLikelihood(data_matrix)

        return log_likelihood


class SCSTClassModel:
    '''
    Models an entire class of vector sequence signatures by extracting conditional distributions
    for vector elements and their relationships to other elements in the same and previous vectors.
    '''

    def __init__(self, train_event_list: List[np.ndarray], max_num_depend: int, mi_thresh: float,
                 min_prob: float, max_value: int, prior_only: bool, num_pri_obs: Optional[int]):
        '''
        :param train_event_list, max_num_depend, mi_thresh, min_prob, max_value:
            See SCSTElementModel.__init__.
        :param prior_only: If True, then only a prior model will be trained, and the dependencies
            between vectors in a sequence won't be exploited by this model.
        :param num_pri_obs: Number of observations at the beginning of each event to use for finding
            prior probabilities. The entirety of each event is used if this value is None.
        '''

        assert num_pri_obs is None or num_pri_obs >= 0
        assert len(train_event_list) > 0

        self._vect_dim = train_event_list[0].shape[0]
        self._previous_vector = None
        self._depend_model = None

        if not prior_only:
            # Assume first order vector dependency for simplicity. A P-order dependency requires P+1
            # models.
            self._depend_model = SCSTVectorModel(train_event_list, 1, max_num_depend, mi_thresh,
                                                min_prob, max_value)

            train_event_list = SCSTClassModel._getPriorEventList(train_event_list, num_pri_obs)

        self._prior_model = SCSTVectorModel(train_event_list, 0, max_num_depend, mi_thresh,
                                            min_prob, max_value)


    def logLikelihood(self, feature_vector: np.ndarray) -> float:
        '''
        Calculate and return the log-likelihood of the input ``feature_vector``, given this model,
        and previous vectors in the same sequence. It is assumed that the vectors processed by this
        method are passed in the order they appear in a sequence, so that dependencies between them
        can be exploited.

        :param feature_vector: The next feature vector in the sequence.

        :return: The log-likelihood of the input vector ``feature_vector``.
        '''

        assert len(feature_vector) == self._vect_dim

        feature_vector = feature_vector.reshape((self._vect_dim, 1))

        if self._previous_vector is None:
            log_likelihood = self._prior_model.logLikelihood(feature_vector)
        else:
            log_likelihood = self._depend_model.logLikelihood(
                np.concatenate((self._previous_vector, feature_vector), axis=1))

        if self._depend_model is not None:
            # Only keep the previous vector if this isn't a prior-only model
            self._previous_vector = feature_vector

        return log_likelihood

    def reset(self) -> None:
        '''
        Reset this model by forcing it to use its prior distribution for calculating the next
        log-likelihood.
        '''
        self._previous_vector = None

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


class SCSTClassifier:
    '''
    Applies a set of ``SCSTClassModel`` to a sequence of vectors (one at a time) to assign class
    labels to each vector as it is processed.
    '''

    def __init__(self, train_event_dict: Dict[Any, List[np.ndarray]], noise_train_array: np.ndarray,
                 quantizer_type: Type[Quantizer], num_quantize_levels: int, max_num_depend: int,
                 mi_thresh: float, min_prob: float, num_pri_obs: Optional[int]):
        '''
        :param train_event_dict: Keys are class labels, and values are lists of training event
            (column vector, sequences) to use for each corresponding class.
        :param noise_train_array: A matrix with column vectors that are samples from noise alone
            (no signatures of any class present).
        :param quantizer_type: Name of quantizer class to use for discretizing vector elements
            prior to classification.
        :param num_quantize_levels: Number of quantization levels to use when training each vector
            element quantizer.
        :param max_num_depend, mi_thresh, and min_prob: See SCSTElementModel.__init__.
        :param num_pri_obs: See SCSTClassModel.__init__.
        '''

        assert len(train_event_dict) > 0
        assert 0 not in train_event_dict, "Label 0 reserved for noise-alone hypothesis"
        assert num_quantize_levels > 0

        class_train_list = next(iter(train_event_dict.values()))
        self._vect_dim = class_train_list[0].shape[0]

        self._quantizer_list = self._getQuantizerList(train_event_dict, noise_train_array,
                                                      quantizer_type, num_quantize_levels)

        self._model_dict, self._noise_model = self._getClassModels(
            train_event_dict, noise_train_array, num_quantize_levels-1, max_num_depend, mi_thresh,
            min_prob, num_pri_obs)

        self.reset()

    def reset(self) -> None:
        '''
        Called when the user has finished analyzing a given sequence and would like to analyze a
        new one, with reinitialized test statistics.
        '''

        # See properties with the same names for an explanation on the purpose of each member.
        self._class_labels = []
        self._signal_llrs = defaultdict(list)
        self._signal_updates = defaultdict(list)
        self._signal_likelihoods = defaultdict(list)
        self._noise_llrs = defaultdict(list)
        self._noise_updates = defaultdict(list)
        self._noise_likelihoods = []

        # Ensure all models are reinitialized to use their priors
        for model in self._model_dict.values():
            model.reset()

        self._detecting_signal = True

    def classify(self, feature_vector: np.ndarray, sig_thresh: float, noise_thresh: float) -> Any:
        '''
        Applies a set of SCSTClassModel to the input ``feature_vector`` to generate and return a
        class label. Class labels are assigned based on cumulative log-likelihood ratios for each
        model, meaning this is a sequential process where the test statistics used to assign
        class labels are constantly updated as new vectors are processed.

        :param feature_vector: The next feature vector in the sequence.
        :param sig_thresh: Cumulative LLR threshold, above which a signal is detected during a
            quiescent period.
        :param noise_thresh: Cumulative inverse LLR threshold, above which noise-only is declared
            during a signal period.

        :return: The class label assigned to ``feature_vector``, which will be one of the keys of
            the dictionary in the constructor input ``train_event_dict``.
        '''

        assert len(feature_vector) == self._vect_dim

        feature_vector = self._quantizeFeatureVector(feature_vector)
        max_likelihood_class = self._updateLikelihoods(feature_vector)

        if self._detecting_signal:
            if self._signal_llrs[max_likelihood_class][-1] > sig_thresh:
                self._detecting_signal = False

        elif self._noise_llrs[max_likelihood_class][-1] > noise_thresh:

            self._detecting_signal = True

            # Reset the LLR test statistics and look for prior state vectors again
            for class_label in self._model_dict.keys():
                self._model_dict[class_label].reset()
                self._signal_llrs[class_label][-1] = 0.0

        if self._detecting_signal:
            self._class_labels.append(0)  # Noise alone hypothesis accepted
        else:
            self._class_labels.append(max_likelihood_class)  # Signal hypothesis accepted

        return self._class_labels[-1]

    @property
    def class_labels(self) -> List[Any]:
        '''
        A list of the class labels assigned to all the vectors processed thus far.
        '''
        return copy.copy(self._class_labels)

    @property
    def signal_llrs(self) -> Dict[Any, float]:
        '''
        Cumulative LLR used to detect signals (keyed by class label).
        '''
        return dict(self._signal_llrs)

    @property
    def signal_updates(self) -> Dict[Any, float]:
        '''
        Statistic used to update corresponding self._sig_llr_list at the same time (keyed by class
        label).
        '''
        return dict(self._signal_updates)

    @property
    def signal_likelihoods(self) -> Dict[Any, float]:
        '''
        Log-likelihood of each signal model. Used with self.noise_likelihoods to calculate the
        corresponding self.sig_updates at the same time (keyed by class label).
        '''
        return dict(self._signal_likelihoods)

    @property
    def noise_llrs(self) -> Dict[Any, float]:
        '''
        Cumulative inverse LLR used to detect noise (keyed by class label).
        '''
        return dict(self._noise_llrs)

    @property
    def noise_updates(self) -> Dict[Any, float]:
        '''
        Statistic used to update corresponding self.noise_llr_list at the same time (keyed by class
        label).
        '''
        return dict(self._noise_updates)

    @property
    def noise_likelihoods(self) -> List[float]:
        '''
        Log-likelihood of the noise model. Used with self.sig_updates to calculate the
        corresponding self.noise_updates at the same time (keyed by class label).
        '''
        return copy.copy(self._noise_likelihoods)

    def _updateLikelihoods(self, feature_vector: np.ndarray) -> Any:
        '''
        Uses the input feature vector to update all the test statistics for every model in this
        classifier.

        :return: The label of the class that has the highest cumulative LLR.
        '''

        noise_ll = self._noise_model.logLikelihood(feature_vector)
        self._noise_likelihoods.append(noise_ll)

        max_likelihood_class = None
        max_signal_llr = -np.inf
        for class_label, model in self._model_dict.items():

            # Update the LLR for each class, storing the statistics along the way so the user can
            # analyze them, if desired.
            signal_ll = model.logLikelihood(feature_vector)
            self._signal_likelihoods[class_label].append(signal_ll)

            signal_update = signal_ll - noise_ll
            self._signal_updates[class_label].append(signal_update)

            # Implement the CUSUM approach to likelihood accumulation, which resets the LLR when it
            # falls below zero.
            signal_llr_list = self._signal_llrs[class_label]
            prev_llr = 0 if not signal_llr_list else signal_llr_list[-1]
            signal_llr = prev_llr + signal_update
            if signal_llr < 0:
                signal_llr = 0
                # LLR is low enough that we should start looking for the beginning of the signal
                # associated with this model (use priors)
                model.reset()

            signal_llr_list.append(signal_llr)

            if signal_llr > max_signal_llr:
                max_signal_llr = signal_llr
                max_likelihood_class = class_label

            # Do the same for noise
            noise_update = noise_ll - signal_ll
            self._noise_updates[class_label].append(noise_update)

            noise_llr_list = self._noise_llrs[class_label]
            if self._detecting_signal:
                # No need to track the noise LLR at this time
                noise_llr_list.append(0.0)
            else:
                prev_llr = 0 if not noise_llr_list else noise_llr_list[-1]
                noise_llr = max(0, prev_llr + noise_update)
                noise_llr_list.append(noise_llr)

        return max_likelihood_class

    def _quantizeFeatureVector(self, feature_vector: np.ndarray) -> np.ndarray:
        '''
        Apply the elements of ``self._quantizer_list`` to the respective elements of
        ``feature_vector`` to generate a discretized feature vector.
        '''

        quantized_vector = np.zeros(self._vect_dim)
        for element_idx, quantizer in enumerate(self._quantizer_list):
            quantized_vector[element_idx] = quantizer(feature_vector[element_idx])

        return quantized_vector

    def _getQuantizerList(self, train_event_dict: Dict[Any, List[np.ndarray]],
                          noise_train_array: np.ndarray, quantizer_type: Type[Quantizer],
                          num_quantize_levels: int) -> List[Quantizer]:
        '''
        Generate and return a list of ``Quantizer`` using the input parameters and dictionary of
        training events. One quantizer is generated for every vector element.
        '''

        # Quantizers are generated using the aggregate of all training data (they do not depend on
        # class label).
        all_feat_matrix = np.array([]).reshape((self._vect_dim, 0))
        for event_list in train_event_dict.values():
            for event in event_list:
                all_feat_matrix = np.concatenate((all_feat_matrix, event), axis=1)

        all_feat_matrix = np.concatenate((all_feat_matrix, noise_train_array), axis=1)

        quantizer_list = []
        for element_idx in range(self._vect_dim):
            quantizer = quantizer_type(num_quantize_levels, all_feat_matrix[element_idx, :], True)
            quantizer_list.append(quantizer)

        return quantizer_list

    def _getClassModels(self, train_event_dict: Dict[Any, List[np.ndarray]],
                        noise_train_array: np.ndarray, max_value: int, max_num_depend: int,
                        mi_thresh: float, min_prob: float,
                        num_pri_obs: Optional[int]) -> Tuple[Dict[Any, SCSTClassModel],
                                                             SCSTClassModel]:
        '''
        Trains a SCSTClassModel for each key in the input ``train_event_dict``, as well as a
        SCSTClassModel for noise-only vector sequences.
        '''

        quantized_train_List = self._getQuantizedTrainEventList([noise_train_array])
        noise_model = SCSTClassModel(quantized_train_List, max_num_depend, mi_thresh, min_prob,
                                     max_value, True, None)

        model_dict = {}
        for class_label, train_event_list in train_event_dict.items():
            quantized_train_List = self._getQuantizedTrainEventList(train_event_list)
            model = SCSTClassModel(quantized_train_List, max_num_depend, mi_thresh, min_prob,
                                   max_value, False, num_pri_obs)
            model_dict[class_label] = model

        return model_dict, noise_model

    def _getQuantizedTrainEventList(self, train_event_list: List[np.ndarray]) -> List[np.ndarray]:
        '''
        Given a list of training events (matrices with column vectors as observations), return a
        list of matrices with the same sizes, but with elements quantized using
        ``self._quantizer_list``.
        '''

        quantized_event_list = []
        for event in train_event_list:

            quantized_event = np.zeros(event.shape)
            for element_idx, quantizer in enumerate(self._quantizer_list):
                quantized_event[element_idx, :] = quantizer(event[element_idx, :])

            quantized_event_list.append(quantized_event)

        return quantized_event_list
