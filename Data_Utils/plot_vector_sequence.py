from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

from read_text_data import getSPLMatrix, getTruthArray, MAX_SECONDS


def _plotIDStrip(ax: matplotlib.axes, id_array: np.ndarray, label: str,
                 x_extents: List[float]) -> None:

    id_array = np.reshape(id_array, (1, len(id_array)))
    ax.imshow(id_array, aspect="auto", interpolation="none",
              extent=x_extents + [0, 1])
    ax.set_ylabel(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])


def plotSequence(data_matrix: np.ndarray, start_second: int, file_name: str,
                 truth_array: Optional[np.ndarray]=None,
                 est_array: Optional[np.ndarray]=None) -> None:
    '''
    Plot 1/3 octave SPL data along with (optionally) estimated and true classification labels.

    :param data_matrix: Matrix with rows that represent 1/3 octave index and columns that represent
        vectors at different times (in seconds).
    :param truth_array: Array with values that represent the true class labels of the columns
        vectors in data_matrix.
    :param est_array: Array with values that represent the estimated class labels of the columns
        vectors in data_matrix.
    :param start_second: Starting time (in seconds) of the input data, relative to midnight.
    :param file_name: Name of the data set to plot (used to generate title).
    '''

    # Dynamically adjust the number of rows based on whether or not we're plotting truth and
    # estimated labels.
    num_rows = 2
    height_ratios = [0.5]  # First row is for the title

    if truth_array is not None:
        assert data_matrix.shape[1] == len(truth_array)
        num_rows += 1
        height_ratios.append(1)

    if est_array is not None:
        assert data_matrix.shape[1] == len(est_array)
        num_rows += 1
        height_ratios.append(1)

    # Setup plot
    fig = plt.figure(figsize=(16, 5), constrained_layout=True)
    fig.suptitle(file_name)
    height_ratios.append(6)
    grid_spec = fig.add_gridspec(ncols=1, nrows=num_rows, height_ratios=height_ratios)

    x_extents = [start_second, start_second + data_matrix.shape[1]]
    current_row = 1
    ax = None

    # Plot estimated class labels
    if est_array is not None:
        ax = fig.add_subplot(grid_spec[current_row, 0])
        _plotIDStrip(ax, est_array, "Est. ID", x_extents)
        current_row += 1

    # Plot true class labels
    if truth_array is not None:
        ax = fig.add_subplot(grid_spec[current_row, 0], sharex=ax)
        _plotIDStrip(ax, truth_array, "Truth ID", x_extents)
        current_row += 1

    # Plot data matrix
    ax = fig.add_subplot(grid_spec[current_row, 0], sharex=ax)
    img = ax.imshow(data_matrix, aspect="auto", interpolation="none", origin="lower",
                    extent=x_extents + [1, data_matrix.shape[0]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("1/3 Octave Index")

    color_bar = fig.colorbar(img, orientation="horizontal")
    color_bar.set_label("SPL (dB)")

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="GRSA001/NVSPL_GRSA001_2008_09_25.txt", type=str)
    parser.add_argument("--truth_path", default="GRSA001/TRUTH_GRSA001_2008_09_25_h16-18.txt",
                        type=str)
    parser.add_argument("--start_second", default=0, type=int)
    parser.add_argument("--end_second", default=MAX_SECONDS, type=int)
    args = parser.parse_args()

    if args.truth_path is not None:
        truth_array, args.start_second, end_second = getTruthArray(args.truth_path)
        data_matrix = getSPLMatrix(args.data_path, args.start_second, end_second)
    else:
        truth_array = None
        data_matrix = getSPLMatrix(args.data_path, args.start_second, args.end_second)

    file_name = args.data_path.split('.txt')[0].split('/')[-1]
    plotSequence(data_matrix, args.start_second, file_name, truth_array, None)
