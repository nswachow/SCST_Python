from typing import Tuple
import numpy as np


VECT_LENGTH = 33
MAX_SECONDS = 86400  # Number of seconds per day


def _processTextLines(file_path: str, start_offset: int, end_offset: int) -> np.ndarray:

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


def getSPLMatrix(file_path: str, start_second: int=0, end_second: int=MAX_SECONDS) -> np.ndarray:
    '''
    Convert data in a 'NVSPL' file to a set of vectors (one vector per line) that represent sound
    pressue levels; in particular, the average energy in different 1/3 octave frequency bands. It
    is assuemd that each file contains measurements every second for an entire day.

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

    assert data_matrix.shape[0] == VECT_LENGTH, "Unexpected vector length: {}.".format(data_matrix.shape[0])
    if data_matrix.shape[1] != end_second - start_second:
        print("Warning: only extracted {} seconds of data.".format(data_matrix.shape[1]))

    return data_matrix


def getTruthArray(file_path: str) -> Tuple[np.ndarray, int, int]:
    '''
    Convert data in a 'TRUTH' file to an array (one element per line). It is assuemd that each file
    follows the format *h<start_hour>-<end_hour>.txt, in order to properly parse the times
    represented by the file.

    :param file_path: Absolute path to the TRUTH file.

    :return: Tuple with the following elements:
        - Array of truth labels
        - Starting time (in seconds) associated with the returned array.
        - Ending time (in seconds) associated with the returned array.
    '''

    hour_strings = file_path.split('.')[0].split('h')[-1].split('-')
    start_second, end_second = (3600 * float(val) for val in hour_strings)
    truth_array = _processTextLines(file_path, 0, end_second - start_second).flatten()

    if len(truth_array) != end_second - start_second:
        print("Warning: only extracted {} seconds of truth data.".format(len(truth_array)))

    return truth_array, start_second, end_second
