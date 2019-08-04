from typing import Tuple, Dict, Union, List
import os
import numpy as np
from collections import defaultdict


VECT_DIM = 33
MAX_SECONDS = 86400  # Number of seconds per day


def _processTextLines(file_path: str, start_offset: int,
                      end_offset: Union[int, float]) -> np.ndarray:
    '''
    Read numerical values from a text file and return an array with the lines in the file as rows.
    Naturally, this assumes that each line in the file has the same number of entries (separated by
    a space).
    '''

    array_list = []
    with open(file_path, 'r') as spl_file:
        for file_idx, file_line in enumerate(spl_file):

            # Since there are no time stamps, assume the first line is the measurement at time 0
            if file_idx < start_offset:
                continue
            elif file_idx >= end_offset:
                break

            data_array = np.array([float(val) for val in file_line.split(' ')])
            array_list.append(data_array)

    return np.array(array_list).T


def _getLabel(original_label: int, convert_interference_to_noise: bool) -> int:
    '''
    Returns a noise label for interfence and noise, if ``convert_interference_to_noise`` is True.
    Returns the original label otherwise. The original label is always returned for signal labels.
    '''

    if ((original_label % 10 != 0) or  # signal labels end in '1'
        (not convert_interference_to_noise)):
        return original_label

    return 0


def getSPLMatrix(file_path: str, start_second: int=0, end_second: int=MAX_SECONDS) -> np.ndarray:
    '''
    Convert data in a 'NVSPL' file to a set of vectors (one vector per line) that represent sound
    pressue levels; in particular, the average energy in different 1/3 octave frequency bands. It
    is assumed that each file contains measurements every second for an entire day.

    :param file_path: Absolute path to the NVSPL file.
    :param start_second: The second to start extracting data (inclusive), where 0 represents
        midnight on the day represented by the input file.
    :param end_second: The second to finish extracting data (exclusive).

    :return: Matrix with data vectors as columns.
    '''

    assert start_second >=0
    assert end_second <= MAX_SECONDS, "end_second must be <= the number of seconds in a day"
    assert end_second > start_second

    data_matrix = _processTextLines(file_path, start_second, end_second)

    assert data_matrix.shape[0] == VECT_DIM, (
        "Unexpected vector length: {}.".format(data_matrix.shape[0]))

    if data_matrix.shape[1] != end_second - start_second:
        print("Warning: only extracted {} seconds of data.".format(data_matrix.shape[1]))

    return data_matrix


def getTruthArray(file_path: str) -> Tuple[np.ndarray, int, int]:
    '''
    Convert data in a 'TRUTH' file to an array (one element per line). It is assumed that each file
    follows the format *h<start_hour>-<end_hour>.txt, in order to properly parse the times
    represented by the file.

    :param file_path: Absolute path to the TRUTH file.

    :return: Tuple with the following elements:
        - Array of truth labels.
        - Starting time (in seconds) associated with the returned array.
        - Ending time (in seconds) associated with the returned array.
    '''

    file_name = os.path.basename(file_path)
    hour_strings = file_name.split(".txt")[0].split('h')[-1].split('-')
    start_second, end_second = (3600 * float(val) for val in hour_strings)
    truth_array = _processTextLines(file_path, 0, end_second - start_second).flatten().astype(int)

    if len(truth_array) != end_second - start_second:
        print("Warning: only extracted {} seconds of truth data.".format(len(truth_array)))

    return truth_array, start_second, end_second


def getTrainEvents(train_file_path: str, data_file_path: str,
                   convert_interference_to_noise: bool) -> Dict[int, List[np.ndarray]]:
    '''
    Extracts events of intrest (typically to use for classifier training), each consisting of a
    sequence of vectors (columns of a matrix), from a vector sequence stored at ``data_file_path``,
    using the events parameters stored in ``train_file_path``.

    :param train_file_path: Absolute path to the file containing training event parameters.
    :param data_file_path: Absolute path to the NVSPL data file.

    :return: Dictionary with keys that are class labels and values that are lists of matrices
        representing events associated with that class.
    '''

    data_matrix = getSPLMatrix(data_file_path)
    train_param_matrix = _processTextLines(train_file_path, 0, np.inf).astype(int)

    event_dict = defaultdict(list)
    for event_idx in range(train_param_matrix.shape[1]):
        start_time, end_time, label = train_param_matrix[:, event_idx]
        event_matrix = data_matrix[:, start_time:end_time+1]
        label = _getLabel(label, convert_interference_to_noise)
        event_dict[label].append(event_matrix)

    return dict(event_dict)
