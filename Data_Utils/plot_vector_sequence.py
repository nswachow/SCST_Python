from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from read_nvspl import getSPLMatrix


def _plotIDStrip(ax: matplotlib.axes, id_array: np.ndarray, label: str, x_extents: List[float]):

    ax.imshow(id_array, aspect="auto", interpolation="none",
              extent=x_extents + [0, 1])
    ax.set_ylabel(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])


def plotSequence(data_matrix: np.ndarray, truth_array: np.ndarray, est_array: np.ndarray,
                 start_second: int, file_name: str):
    '''
    Plot 1/3 octave SPL data along with (optionall) estimated and true classification labels.

    :param data_matrix: Matrix with rows that represent 1/3 octave index and columns that represent
        vectors at different times (in seconds).
    :param truth_array: Array with values that represent the true class labels of the columns
        vectors in data_matrix.
    :param est_array: Array with values that represent the estimated class labels of the columns
        vectors in data_matrix.
    :param start_second: Starting time (in seconds) of the input data, relative to midnight.
    :param file_name: Name of the data set to plot (used to generate title).
    '''

    assert data_matrix.shape[1] == truth_array.shape[1]
    assert data_matrix.shape[1] == est_array.shape[1]

    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    fig.suptitle(file_name)
    grid_spec = fig.add_gridspec(ncols=1, nrows=4, height_ratios=[0.5, 1, 1, 6])

    x_extents = [start_second, start_second + data_matrix.shape[1]]
    ax = fig.add_subplot(grid_spec[1, 0])
    _plotIDStrip(ax, est_array, "Est. ID", x_extents)

    ax = fig.add_subplot(grid_spec[2, 0], sharex=ax)
    _plotIDStrip(ax, truth_array, "Truth ID", x_extents)

    ax = fig.add_subplot(grid_spec[3, 0], sharex=ax)
    img = ax.imshow(data_matrix, aspect="auto", interpolation="none", origin="lower",
                    extent=x_extents + [1, data_matrix.shape[0]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("1/3 Octave Index")

    color_bar = fig.colorbar(img, orientation="horizontal")
    color_bar.set_label("SPL (dB)")

    plt.show()


if __name__ == '__main__':

    FILE_PATH = "GRSA001/NVSPL_GRSA001_2008_09_25.txt"
    START_SECOND = 1800
    END_SECOND = 3600

    data_matrix = getSPLMatrix(FILE_PATH, START_SECOND, END_SECOND)
    fake_est = np.reshape(np.random.randint(low=0, high=3, size=1800), (1, 1800))
    fake_truth = np.reshape(np.random.randint(low=0, high=3, size=1800), (1, 1800))

    file_name = FILE_PATH.split('.txt')[0].split('/')[-1]

    plotSequence(data_matrix, fake_truth, fake_est, START_SECOND, file_name)
