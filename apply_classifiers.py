import os
from pathlib import Path
import argparse
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt

from scst.scst_model import SCSTClassifier
from data_utils.read_text_data import getTruthArray, getSPLMatrix
from data_utils.plot_vector_sequence import plotSequence, getPlotDataTuple


def applyClassifiers(scst_pkl_path: str, data_path: Optional[str]=None,
                     truth_path: Optional[str]=None, start_second: Optional[float]=None,
                     end_second: Optional[float]=None) -> None:
    '''
    Apply a SCSTClassifier to the data specified by the inputs, and plot the results.

    :param scst_pkl_path: Full path to a pickle file containing a saved SCSTClassifier object.
    :param data_path, truth_path, start_second, end_second: See ``data_utils.getPlotDataTuple``.
    '''

    SIG_THRESH = 100
    NOISE_THRESH = 100

    test_matrix, truth_array, start_second, title = getPlotDataTuple(
        truth_path, data_path, start_second, end_second)
    classifier = SCSTClassifier.load(scst_pkl_path)

    for test_idx in range(test_matrix.shape[1]):
        classifier.classify(test_matrix[:, test_idx], SIG_THRESH, NOISE_THRESH)

    print("CC Rate:", np.sum(classifier.class_labels == truth_array) / len(truth_array))

    plotSequence(test_matrix, start_second, title, [truth_array, classifier.class_labels],
                 ["Truth IDS", "SCST IDs"])

    plt.figure()
    for label in classifier._model_dict.keys():
        if label != 0:
            plt.plot(classifier.signal_llrs[label])

    plt.show()


if __name__ == '__main__':

    this_dir = os.path.dirname(Path(__file__).absolute())
    data_dir = os.path.join(this_dir, "GRSA001")

    default_scst_pkl_path = os.path.join(this_dir, "scst_classifier.pkl")
    default_truth_path = os.path.join(data_dir, "TRUTH_GRSA001_2008_10_03_h21-23.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--scst_pkl_path", default=default_scst_pkl_path)
    parser.add_argument("--truth_path", default=default_truth_path, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--start_second", default=None, type=int)
    parser.add_argument("--end_second", default=None, type=int)
    args = parser.parse_args()

    applyClassifiers(args.scst_pkl_path, args.data_path, args.truth_path, args.start_second,
                     args.end_second)
