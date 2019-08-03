import os
from pathlib import Path

from scst.scst_model import SCSTClassifier
from scst.quantizer import LloydMaxQuantizer
from data_utils.read_text_data import getTrainEvents


def trainSCSTModel() -> None:
    '''
    Train and save a new SCSTModel using events and parameters specified within.
    '''

    # Define training parameters
    SAVE_NAME = "scst_classifier.pkl"
    TRAIN_DATE = "2008_10_02"
    MAX_NUM_DEPEND = 1
    MI_THRESH = 0.05
    MIN_PROB = 1e-4
    NUM_PRI_OBS = 10
    NUM_QUANTIZE_LEVELS = 5

    # Get training data
    this_dir = os.path.dirname(Path(__file__).absolute())
    data_dir = os.path.join(this_dir, "GRSA001")
    train_file_name = "TRAIN_GRSA001_" + TRAIN_DATE + ".txt"
    train_file_path = os.path.join(data_dir, train_file_name)

    data_file_name = "NVSPL_GRSA001_" + TRAIN_DATE + ".txt"
    data_file_path = os.path.join(data_dir, data_file_name)

    train_event_dict = getTrainEvents(train_file_path, data_file_path, True)

    # Train and save classifier
    classifier = SCSTClassifier(train_event_dict, LloydMaxQuantizer, NUM_QUANTIZE_LEVELS,
                                MAX_NUM_DEPEND, MI_THRESH, MIN_PROB, NUM_PRI_OBS)
    save_path = os.path.join(this_dir, SAVE_NAME)
    classifier.save(save_path)


if __name__ == '__main__':
    trainSCSTModel()
