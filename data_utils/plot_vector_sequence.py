import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import argparse

from data_utils.read_text_data import getSPLMatrix, getTruthArray


def _plotIDStrip(ax: matplotlib.axes, id_array: np.ndarray, label: str,
                 x_extents: List[float]) -> None:
    '''
    Plot a strip above the data that represents class labels at each time.
    '''

    id_array = np.reshape(id_array, (1, len(id_array)))
    ax.imshow(id_array, aspect="auto", interpolation="none",
              extent=x_extents + [0, 1])
    ax.set_ylabel(label)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_ticks([])


def getPlotDataTuple(truth_path: Optional[str],
                     data_path: Optional[str],
                     start_second: Optional[int],
                     end_second: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray],
                                                         float, str]:
    '''
    Retrieve either:
    1) A data matrix (feature vectors are columns) based on the inputs data_path, start_second,
        end_second.
    2) An array with truth labels and a corresponding matrix of data (covers the same time segment),
        based on the input train_path.
    ... plus the start time of the sequence, and a title for the plot.

    :param truth_path: The full path to a "TRUTH" file that describes the location and types of
        events.
    :param data_path: The full path to a "NVSPL" file that contains raw 1/3 data. If this input is
        not ``None`` then the ``truth_path`` input will be ignored.
    :param start_second: The starting time of the data to extract from the data at ``data_path``.
        This value is irrelevant when using the ``truth_path`` input instead of the ``data_path``
        input.
    :param end_second: The ending time of the data to extract from the data at ``data_path``. This
        value is irrelevant when using the ``truth_path`` input instead of the ``data_path`` input.

    :return: See method description.
    '''

    if data_path is not None:
        assert (start_second is not None) and (end_second is not None), (
            "Must supply a start_second and end_second with data_path")
        truth_array = None
        base_name = os.path.basename(data_path).split(".txt")[0].split("NVSPL_")[1]
        data_matrix = getSPLMatrix(data_path, start_second, end_second)

    else:
        assert truth_path is not None, "Must supply truth_path if no data_path is supplied"

        truth_dir = os.path.dirname(truth_path)
        base_name = os.path.basename(truth_path).split("_h")[0].split("TRUTH_")[1]
        data_path = os.path.join(truth_dir, "NVSPL_" + base_name + ".txt")

        truth_array, start_second, end_second = getTruthArray(truth_path)
        data_matrix = getSPLMatrix(data_path, start_second, end_second)

    return data_matrix, truth_array, start_second, base_name


def plotSequence(data_matrix: np.ndarray, start_second: int, title: str,
                 id_array_list: Optional[List[np.ndarray]]=None,
                 id_tag_list: Optional[List[str]]=None) -> None:
    '''
    Plot 1/3 octave SPL data along with (optionally) estimated and true classification labels.

    :param data_matrix: Matrix with rows that represent 1/3 octave index and columns that represent
        vectors at different times (in seconds).
    :param start_second: Starting time (in seconds) of the input data, relative to midnight.
    :param title: Title to place on the plot.
    :param id_array_list: Elements are arrays with equal length that represent class labels
        associated with the corresponding columns of ``data_matrix``.
    :param id_tag_list: Elements are strings used to label a class label array in the corresponding
        element of ``id_array_list``.
    '''

    # Dynamically adjust the number of rows based on whether or not we're plotting truth and
    # estimated labels.
    num_rows = 2
    height_ratios = [0.5]  # First row is for the title
    plot_height = 3

    do_id_strips = (id_array_list is not None) and (id_tag_list is not None)

    if do_id_strips:
        assert len(id_array_list) == len(id_tag_list)
        for id_array in id_array_list:
            assert data_matrix.shape[1] == len(id_array)
            num_rows += 1
            plot_height += 1
            height_ratios.append(1)

    # Setup plot
    fig = plt.figure(figsize=(16, plot_height), constrained_layout=True)
    fig.suptitle(title)
    height_ratios.append(4)
    grid_spec = fig.add_gridspec(ncols=1, nrows=num_rows, height_ratios=height_ratios)

    x_extents = [start_second, start_second + data_matrix.shape[1]]
    current_row = 1
    ax = None

    # Plot ID strips
    if do_id_strips:
        for id_array, id_tag in zip(id_array_list, id_tag_list):
            ax = fig.add_subplot(grid_spec[current_row, 0], sharex=ax)
            _plotIDStrip(ax, id_array, id_tag, x_extents)
            current_row += 1

    # Plot data matrix
    ax = fig.add_subplot(grid_spec[current_row, 0], sharex=ax)
    img = ax.imshow(data_matrix, aspect="auto", interpolation="none", origin="lower",
                    extent=x_extents + [1, data_matrix.shape[0]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("1/3 Octave Index")

    color_bar = fig.colorbar(img, orientation="horizontal")
    color_bar.set_label("SPL (dB)")


if __name__ == '__main__':

    this_dir = os.path.dirname(Path(__file__).absolute())
    data_dir = os.path.join(this_dir, "..", "GRSA001")
    default_truth_path = os.path.join(data_dir, "TRUTH_GRSA001_2008_10_03_h21-23.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument("--truth_path", default=default_truth_path, type=str)
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--start_second", default=None, type=int)
    parser.add_argument("--end_second", default=None, type=int)
    args = parser.parse_args()

    data_matrix, truth_array, start_second, title = getPlotDataTuple(
        args.truth_path, args.data_path, args.start_second, args.end_second)

    plotSequence(data_matrix, start_second, title, [truth_array], ["Truth IDs"])

    plt.show()
