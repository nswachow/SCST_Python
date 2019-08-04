import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

from keras.layers import Dense
from keras.models import Sequential

from scst.scst_model import SCSTClassifier
from scst.quantizer import LloydMaxQuantizer
from transient_keras.transient_keras_classifier import TransientKerasClassifier
from data_utils.read_text_data import getTrainEvents


def _trainSCSTClassifier(train_event_dict: Dict[Any, List[np.ndarray]],
                         save_dir: str) -> None:
    '''
    Train and save a new SCSTModel using the input events and the parameters specified within.
    '''

    # Define training parameters
    SAVE_NAME = "scst_classifier.pkl"
    MAX_NUM_DEPEND = 1
    MI_THRESH = 0.05
    MIN_PROB = 1e-4
    NUM_PRI_OBS = 10
    NUM_QUANTIZE_LEVELS = 5

    # Train and save classifier
    classifier = SCSTClassifier(train_event_dict, LloydMaxQuantizer, NUM_QUANTIZE_LEVELS,
                                MAX_NUM_DEPEND, MI_THRESH, MIN_PROB, NUM_PRI_OBS)
    save_path = os.path.join(save_dir, SAVE_NAME)
    classifier.save(save_path)


def _trainTransientKerasClassifier(train_event_dict: Dict[Any, List[np.ndarray]],
                                   save_dir: str) -> None:
    '''
    Train and save a new TransientKerasClassifier using the input events and the parameters
    specified within.
    '''

    # Define training parameters
    SAVE_NAME = "keras_classifier.pkl"
    EVAL_PERCENT = 0.33
    NUM_INIT = 5
    NUM_EPOCH = 20
    BATCH_SIZE = 32
    VECT_DIM = 33

    # Found via "fine_best_keras_model.py" script
    FIRST_LAYER_NEURONS = 60
    SECOND_LAYER_NEURONS = 28

    # Train and save classifier
    num_classes = len(train_event_dict)

    model = Sequential()
    model.add(Dense(FIRST_LAYER_NEURONS, activation='relu', input_dim=2*VECT_DIM))
    if SECOND_LAYER_NEURONS > 0:
        model.add(Dense(SECOND_LAYER_NEURONS, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model_config = model.get_config()

    classifier = TransientKerasClassifier(train_event_dict, model_config, EVAL_PERCENT,
                                          NUM_INIT, NUM_EPOCH, BATCH_SIZE, 0)
    save_path = os.path.join(save_dir, SAVE_NAME)
    classifier.save(save_path)


def trainClassifiers() -> None:
    '''
    Load traininng data and pass it to functions to train an SCSTClassifier and a
    TransientKerasClassifier.
    '''

    TRAIN_DATE = "2008_10_02"

    # Get training data
    this_dir = os.path.dirname(Path(__file__).absolute())
    data_dir = os.path.join(this_dir, "GRSA001")
    train_file_name = "TRAIN_GRSA001_" + TRAIN_DATE + ".txt"
    train_file_path = os.path.join(data_dir, train_file_name)

    data_file_name = "NVSPL_GRSA001_" + TRAIN_DATE + ".txt"
    data_file_path = os.path.join(data_dir, data_file_name)
    train_event_dict = getTrainEvents(train_file_path, data_file_path, True)

    _trainSCSTClassifier(train_event_dict, this_dir)
    _trainTransientKerasClassifier(train_event_dict, this_dir)


if __name__ == '__main__':
    trainClassifiers()
