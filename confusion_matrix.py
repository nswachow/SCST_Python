import numpy as np
from typing import Tuple, Any, List
import matplotlib.pyplot as plt
import seaborn


class ConfusionMatrix:
    '''
    Analyzes labels assigned to feature vectors by a classifier, relative to true class labels, in
    order to generate a matrix helpful for identifying the underlying reasons behind classification
    results.
    '''

    def __init__(self, assigned_array: np.ndarray, truth_array: np.ndarray, use_percent: bool,
                 classifier_name: str):

        self._use_percent = use_percent
        self._classifier_name = classifier_name
        self._matrix, self._labels, correct_rate = self._getConfusionMatrix(assigned_array,
                                                                            truth_array)
        self._correct_rate_string = str(int(1e4 * correct_rate) / 1e4)

    def __str__(self):
        '''
        Return a string representation of this confusion matrix.
        '''

        np.set_printoptions(suppress=True)
        header = (self._classifier_name + " Correct Classification Rate: " +
                  self._correct_rate_string +
                  "\nConfusion Matrix (Class Labels: " + str(self._labels) +
                  ";  Rows = truth, Columns = assigned)\n")
        return header + str(self._matrix) + '\n'

    def _getConfusionMatrix(self, assigned_array: np.ndarray,
                            truth_array: np.ndarray) -> Tuple[np.ndarray, List[Any], float]:
        '''
        Calculate and return the confusion matrix, along with a list of unqiue class labels, and the
        correct classification rate.
        '''

        # Create a dictionary for fast lookup of matrix indicies
        unique_labels = list(set(np.append(assigned_array, truth_array)))
        label_idx_dict = {}
        for idx, label in enumerate(unique_labels):
            label_idx_dict[label] = idx

        # Iterate through the label arrays to populate the various elements of the matrix
        num_classes = len(unique_labels)
        conf_matrix = np.zeros((num_classes, num_classes))
        for assigned, truth in zip(assigned_array, truth_array):
            conf_matrix[label_idx_dict[truth], label_idx_dict[assigned]] += 1

        # Calculate correct classification rate
        correct_rate = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

        # Normalize so matrix represents percentages, if applicable
        if self._use_percent:
            conf_matrix /= np.sum(conf_matrix)
            for row_idx in range(conf_matrix.shape[0]):
                row_sum = np.sum(conf_matrix[row_idx, :])
                if row_sum > 0.0:
                    conf_matrix[row_idx, :] = 100 * conf_matrix[row_idx, :] / row_sum

        return conf_matrix, unique_labels, correct_rate

    def display(self) -> None:
        '''
        Use seaborn to display the confusion matrix.
        '''

        plt.figure()
        color_bar_label = "Percent" if self._use_percent else "Occurrences"
        seaborn.heatmap(self._matrix, annot=True, cbar_kws={'label': color_bar_label}, fmt='g',
                        xticklabels=self._labels, yticklabels=self._labels)

        plt.xlabel("Assigned Class")
        plt.ylabel("True Class")
        plt.title(self._classifier_name + " Confusion Matrix (Correct Classification Rate: " +
                  self._correct_rate_string + ")")
