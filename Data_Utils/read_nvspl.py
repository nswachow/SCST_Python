import numpy as np


def getSPLMatrix(file_path: str, start_second: int=0, end_second: int=86400):
    '''
    Convert data in a 'NVSPL' file to a set of vectors (one vector per line), and return a matrix with these vectors as columns. It is assuemd that each file contains measurements every second
    for an entire day.

    :param file_path: Absolute path to the NVSPL file.
    :param start_time: The second to start extracting data (inclusive), where 0 represents midnight
        on the day represented by the input file.
    :param end_second: The second to finish extracting data (exclusive).
    '''

    VECT_LENGTH = 33
    MAX_SECONDS = 86400  # Number of seconds per day

    assert start_second >=0
    assert end_second <= MAX_SECONDS, "end_second must be <= the number of seconds in a day"
    assert end_second > start_second

    vector_list = []
    with open(file_path, 'r') as spl_file:
        for file_idx, file_line in enumerate(spl_file):

            # Since there are no time stamps, assume the first line is the measurement at time 0
            if file_idx < start_second:
                continue
            elif file_idx >= end_second:
                break

            oto_vect = np.array([float(val) for val in file_line.split(' ')])
            assert len(oto_vect) == VECT_LENGTH
            vector_list.append(oto_vect)

    return np.array(vector_list).T
