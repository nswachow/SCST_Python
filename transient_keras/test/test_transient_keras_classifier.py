import numpy as np
from keras.models import Sequential
from keras.layers import Dense

from transient_keras.transient_keras_classifier import TransientKerasClassifier
from test_tools import getTrainDict, getTestSequence


def testKerasSequenceClassifier():

    # Define parameters
    EVAL_PERCENT = 0.33
    NUM_INIT = 2
    NUM_EPOCH = 10
    BATCH_SIZE = 32

    OBS_PER_EVENT = 100
    EVENTS_PER_CLASS = 10
    VECT_DIM = 9

    # Retrieve train and test data
    train_event_dict = getTrainDict(OBS_PER_EVENT, EVENTS_PER_CLASS, VECT_DIM)
    test_array, _ = getTestSequence(VECT_DIM)
    num_classes = len(train_event_dict)

    # Initialize classifier and generate results
    model = Sequential()
    model.add(Dense(6, activation='relu', input_dim=2*VECT_DIM))
    model.add(Dense(num_classes, activation='softmax'))
    model_config = model.get_config()

    classifier = TransientKerasClassifier(train_event_dict, model_config, EVAL_PERCENT, NUM_INIT,
                                          NUM_EPOCH, BATCH_SIZE, 0)

    for test_idx in range(test_array.shape[1]):
        classifier.classify(test_array[:, test_idx])

    num_vect = test_array.shape[1]
    assert len(classifier.class_labels) == num_vect

    classifier.reset()
    assert not classifier.class_labels


if __name__ == '__main__':
    testKerasSequenceClassifier()
