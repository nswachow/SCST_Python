import numpy as np
import copy
import pickle
from typing import Tuple, Dict, List, Any

from keras.models import Sequential
import keras.utils


class TransientKerasClassifier():
    '''
    Wraps a keras neural network to generate class labels for sequences of vector data. Provides
    functionality to train the network using a dictionary of event (vector sequence) lists for each
    class, by automatically partitioning the data to be in a keras-compatible format.
    '''

    def __init__(self, train_event_dict: Dict[Any, List[np.ndarray]], model_config: Dict[str, Any],
                 eval_percent: float, num_init: int, num_epoch: int, batch_size: int,
                 default_label: Any):
        '''
        :param train_event_dict: Keys are class labels, and values are lists of training event
            (column vector) sequences to use for each corresponding class.
        :param model_config: Represents the configuration information for a keras model, obtained
            using the <model>.get_config() method.
        :param eval_percent: Percentage of the input training data to use for evaluating each
            initialized network, to facilitate selection of the best performing network among those
            trained.
        :param num_init: Number of different initializations to try, where the highest performing
            resultant network is retained by this object.
        :param num_epoch: Number of epochs to use when training the keras network.
        :param batch_size: Batch size to use when training the keras network.
        :param default_label: Label to return when no previous vectors have been processed by the
            classify method. Needed since the keras network is trained to make decisions on pairs
            of feaure vectors that are temporaly adjacent.
        '''

        self._default_label = default_label
        self._vect_dim = next(iter(train_event_dict.values()))[0].shape[0]

        self._class_label_dict, train_event_dict = TransientKerasClassifier._getClassLabelDict(
            train_event_dict)
        self._model, self.score = self._getKerasModel(train_event_dict, model_config, eval_percent,
                                                      num_init, num_epoch, batch_size)
        self.reset()

        # These input's aren't used hereafter, but store them so they can be referenced from saved
        # models.
        self.model_config = model_config
        self.eval_percent = eval_percent
        self.num_init = num_init
        self.num_epoch = num_epoch
        self.batch_size = batch_size

    @property
    def class_labels(self) -> List[Any]:
        '''
        A list of the class labels assigned to all the vectors processed thus far.
        '''
        return copy.copy(self._class_labels)

    def save(self, file_path: str) -> None:
        '''
        Serialize this object to a pickle file.

        :param file_path: The full path to the file to be used to store this object.
        '''
        with open(file_path, 'wb') as pkl_file:
            pickle.dump(self, pkl_file)

    @staticmethod
    def load(file_path: str) -> 'TransientKerasClassifier':
        '''
        Load and return a ``TransientKerasClassifier`` object located at the input file path.

        :param file_path: The full path to the file storing the object to load.
        '''
        with open(file_path, 'rb') as pkl_file:
            classifier = pickle.load(pkl_file)
            assert isinstance(classifier, TransientKerasClassifier)

        return classifier

    def classify(self, feature_vector: np.ndarray) -> Any:
        '''
        Applies the internal keras network to the input ``feature_vector`` to generate and return a
        class label. Class labels are assigned based on output neurons with the largest response
        to the input.

        :param feature_vector: The next feature vector in the sequence.

        :return: The class label assigned to ``feature_vector``, which will be one of the keys of
            the dictionary in the constructor input ``train_event_dict``.
        '''
        assert len(feature_vector) == self._vect_dim

        if self._previous_vector is None:
            # We can't form an appropriate input vector to the keras network without a previous
            # vector in the sequence.
            self._previous_vector = feature_vector
            self._class_labels.append(self._default_label)

        else:
            input_vector = np.append(feature_vector, self._previous_vector)
            self._previous_vector = feature_vector

            response = self._model.predict(input_vector.reshape((1, 2*self._vect_dim)))
            class_idx = np.argmax(response)
            self._class_labels.append(self._class_label_dict[class_idx])

        return self._class_labels[-1]

    def reset(self) -> None:
        '''
        Reset this model to prepare it for application to a new sequence.
        '''
        self._previous_vector = None
        self._class_labels = []

    @staticmethod
    def _getClassLabelDict(train_event_dict: Dict[Any, List[np.ndarray]]) \
            -> Tuple[Dict[int, Any], Dict[int, List[np.ndarray]]]:
        '''
        Generate and return the following:
        1) A dictionary that provides a mapping between the keys of ``train_event_dict``, which are
        the original class labels, and contiguous integer class labels that keras operates on.
        2) A dictionary containing the same training event list as the input ``train_event_dict``,
        but keyed by the appropriate keras integer class label.
        '''

        int_label = 0
        int_class_label_dict = {}
        int_train_event_dict = {}
        for label, event_list in train_event_dict.items():
            int_class_label_dict[int_label] = label
            int_train_event_dict[int_label] = event_list
            int_label += 1

        return int_class_label_dict, int_train_event_dict

    def _getKerasModel(self, train_event_dict: Dict[int, List[np.ndarray]],
                       model_config: Dict[str, Any], eval_percent: int, num_init: int,
                       num_epoch: int, batch_size: int) -> Tuple[Sequential, float]:
        '''
        Train ``num_init`` keras models according to the input parameters, and return that which
        produces the best score.
        '''

        data_array, class_labels = self._getKerasCompatibleData(train_event_dict)

        score_array = np.zeros(num_init)
        model_list = []
        for idx in range(num_init):
            model, score = TransientKerasClassifier._trainKerasModel(
                model_config, data_array, class_labels, eval_percent, num_epoch, batch_size)
            score_array[idx] = score
            model_list.append(model)

        best_model_idx = np.argmin(score_array)

        return model_list[best_model_idx], score_array[best_model_idx]

    @staticmethod
    def _trainKerasModel(model_config: Dict[str, Any], data_array: np.ndarray,
                         class_labels: np.ndarray, eval_percent: float, num_epoch: int,
                         batch_size: int) -> Tuple[Sequential, Tuple[float]]:
        '''
        Train a single keras model according to the input parameters.
        '''
        train_data, train_labels, eval_data, eval_labels = \
            TransientKerasClassifier._divideTrainData(data_array, class_labels, eval_percent)
        model = Sequential.from_config(model_config)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit(train_data, train_labels, epochs=num_epoch, batch_size=batch_size)
        score = model.evaluate(eval_data, eval_labels, batch_size=batch_size)

        return model, score

    def _getKerasCompatibleData(self,
                                train_event_dict: Dict[int, List[np.ndarray]]) -> Tuple[np.ndarray,
                                                                                        np.ndarray]:
        '''
        Partition the data in the input ``train_event_dict`` to a format that is appropriate for
        training a keras model.
        '''

        num_classes = len(train_event_dict)

        data_array = np.array([]).reshape((0, 2*self._vect_dim))
        class_labels = np.array([])

        for class_label, event_list in train_event_dict.items():
            for event in event_list:
                depend_event = np.concatenate((event[:, 1:], event[:, :-1])).T
                data_array = np.concatenate((data_array, depend_event))
                current_labels = np.array([class_label] * (event.shape[1]-1))
                class_labels = np.append(class_labels, current_labels)

        class_labels = keras.utils.to_categorical(class_labels, num_classes=num_classes)

        return data_array, class_labels

    @staticmethod
    def _divideTrainData(data_array: np.ndarray, class_labels: np.ndarray,
                         eval_percent: float) -> Tuple[np.ndarray, np.ndarray,
                                                       np.ndarray, np.ndarray]:
        '''
        Segment the input data into training and evaluation partitions, and randomize the order to
        provide different training stimulus.
        '''

        # Randomize order of data
        idx_array = np.random.permutation(len(class_labels))
        data_array = data_array[idx_array, :]
        class_labels = class_labels[idx_array]

        # Split data between training and evaluation sets
        num_eval = int(len(class_labels) * eval_percent)
        eval_data = data_array[:num_eval, :]
        eval_labels = class_labels[:num_eval]

        train_data = data_array[num_eval:, :]
        train_labels = class_labels[num_eval:]

        return train_data, train_labels, eval_data, eval_labels
