import os
import sys
from pathlib import Path
import argparse
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from scst.scst_model import SCSTClassifier
from transient_keras.transient_keras_classifier import TransientKerasClassifier
from data_utils.read_text_data import getTruthArray, getSPLMatrix
from data_utils.plot_vector_sequence import plotSequence, getPlotDataTuple


def _printProgress(percent_complete: float, run_name: str, bar_width: int=50,
                   fill_char: str=u"\u25A0") -> None:
    '''
    A simple terminal progress bar.
    '''

    filled_bar = fill_char * int(round(percent_complete * bar_width))
    empty_bar = ' ' * (bar_width - len(filled_bar))
    sys.stdout.write("\rApplying Classifiers to {0}: [{1}] {2}%".format(
        run_name, filled_bar + empty_bar, int(100 * percent_complete)))
    sys.stdout.flush()


def applyClassifiers(scst_pkl_path: str, keras_pkl_path: str, data_path: Optional[str]=None,
                     truth_path: Optional[str]=None, start_second: Optional[float]=None,
                     end_second: Optional[float]=None) -> None:
    '''
    Apply a SCSTClassifier to the data specified by the inputs, and plot the results.

    :param scst_pkl_path: Full path to a pickle file containing a saved ``SCSTClassifierqq object.
    :param keras_pkl_path: Full path to a pickle file containing a saved
        ``TransientKerasClassifier`` object.
    :param data_path, truth_path, start_second, end_second: See ``data_utils.getPlotDataTuple``.
    '''

    # Define parameters
    SIG_THRESH = 100
    NOISE_THRESH = 100

    # Load data and classifiers
    test_matrix, truth_array, start_second, title = getPlotDataTuple(
        truth_path, data_path, start_second, end_second)
    scst_classifier = SCSTClassifier.load(scst_pkl_path)
    keras_classifier = TransientKerasClassifier.load(keras_pkl_path)

    # Apply classifiers
    num_obs = test_matrix.shape[1]
    for test_idx in range(num_obs):
        scst_classifier.classify(test_matrix[:, test_idx], SIG_THRESH, NOISE_THRESH)
        keras_classifier.classify(test_matrix[:, test_idx])
        _printProgress(test_idx / float(num_obs), title)

    _printProgress(1.0, title)
    sys.stdout.write("\n")

    # Display results
    id_array_list = [scst_classifier.class_labels, keras_classifier.class_labels]
    id_tag_list = ["SCST IDs", "Keras IDs"]

    if truth_array is not None:
        print("SCST Correct Classification Rate:",
            np.sum(scst_classifier.class_labels == truth_array) / len(truth_array))
        print("Keras Correct Classification Rate:",
            np.sum(keras_classifier.class_labels == truth_array) / len(truth_array))

        id_array_list.append(truth_array)
        id_tag_list.append("Truth IDs")

    plotSequence(test_matrix, start_second, title, id_array_list, id_tag_list)
    plt.show()


if __name__ == '__main__':

    this_dir = os.path.dirname(Path(__file__).absolute())
    data_dir = os.path.join(this_dir, "GRSA001")

    default_scst_pkl_path = os.path.join(this_dir, "scst_classifier.pkl")
    default_keras_pkl_path = os.path.join(this_dir, "keras_classifier.pkl")
    default_truth_path = os.path.join(data_dir, "TRUTH_GRSA001_2008_10_03_h21-23.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--scst_pkl_path", default=default_scst_pkl_path)
    parser.add_argument("--keras_pkl_path", default=default_keras_pkl_path)
    parser.add_argument("--truth_path", default=default_truth_path, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--start_second", default=None, type=int)
    parser.add_argument("--end_second", default=None, type=int)
    args = parser.parse_args()

    applyClassifiers(args.scst_pkl_path, args.keras_pkl_path, args.data_path, args.truth_path,
                     args.start_second, args.end_second)
