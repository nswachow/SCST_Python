import os
from pathlib import Path
import numpy as np

from keras.layers import Dense
from keras.models import Sequential

from transient_keras.transient_keras_classifier import TransientKerasClassifier
from data_utils.read_text_data import getTrainEvents, VECT_DIM


def findBestKerasModel():
    '''
    Iterate through a set of network structures, train a corresponding ``TransientKerasClassifier``
    for each structure, and note the score achieved by each. Finally, report the scores of all the
    networks in order from best to worst.
    '''

    # Define training parameters
    TRAIN_DATE = "2008_10_02"
    EVAL_PERCENT = 0.33
    NUM_INIT = 5
    NUM_EPOCH = 30
    BATCH_SIZE = 32

    # Get training data
    this_dir = os.path.dirname(Path(__file__).absolute())
    data_dir = os.path.join(this_dir, "GRSA001")
    train_file_name = "TRAIN_GRSA001_" + TRAIN_DATE + ".txt"
    train_file_path = os.path.join(data_dir, train_file_name)

    data_file_name = "NVSPL_GRSA001_" + TRAIN_DATE + ".txt"
    data_file_path = os.path.join(data_dir, data_file_name)

    train_event_dict = getTrainEvents(train_file_path, data_file_path, True)
    num_classes = len(train_event_dict)

    # Find the optimal network architecture
    neuron_tuple_list = []
    score_list = []
    for first_layer_neurons in np.arange(16, 64, 4):
        for second_layer_neurons in np.arange(0, first_layer_neurons, 4):

            model = Sequential()
            model.add(Dense(first_layer_neurons, activation='relu', input_dim=2*VECT_DIM))
            if second_layer_neurons > 0:
                model.add(Dense(second_layer_neurons, activation='relu'))

            model.add(Dense(num_classes, activation='softmax'))
            model_config = model.get_config()

            classifier = TransientKerasClassifier(train_event_dict, model_config, EVAL_PERCENT,
                                                  NUM_INIT, NUM_EPOCH, BATCH_SIZE, 0)

            neuron_tuple_list.append((first_layer_neurons, second_layer_neurons))
            score_list.append(classifier.score)

            print("NEURONS:", neuron_tuple_list[-1], "SCORE:", score_list[-1])

    # Print results
    sort_idx = np.argsort(score_list)
    sorted_score_list = [score_list[idx] for idx in sort_idx]
    sorted_neuron_list = [neuron_tuple_list[idx] for idx in sort_idx]

    print("\n*** Network Scores ***")
    for neuron_tuple, score in zip(sorted_neuron_list, sorted_score_list):
        print("NEURONS:", neuron_tuple, "SCORE:", score)


if __name__ == '__main__':
    findBestKerasModel()
