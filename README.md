# Transient Source Classification Using Sparse Coefficient State Tracking

The code in this repository implements a sparse coefficient state tracking (SCST) approach for classifying transient signals from vector sequences. More specifically, it provides a Python implementation of a simplified version of the approach introduced in the paper:

N. Wachowski, and M.R. Azimi-Sadjadi, "Detection and classification of nonstationary transient signals using sparse approximations and Bayesian networks,'' *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 22, no. 12, pp. 1750-1764, December 2014.

The original implementation of the SCST approach was in MATLAB, and was designed as more of a research tool, as it was part of my Ph.D. work.


## Background

The SCST method performs multi-class transient classification by processing vectors in a sequence, one at a time. The original formulation applies a sparse decomposition framework to each vector and quantizes the resulting atom coefficients to provide a data representation that can be more readily modeled by simple conditional categorical discrete distributions. This allows for realistically forming the joint likelihoods of different class model parameters, given complicated, high-dimensional vector data. Moreover, since atoms are typically chosen to sparsely represent different signal and interference classes, the signatures of these two categories of coefficients are often separable in the sparse domain, allowing for nullifying the interference coefficients for improved signal detection.

The SCST method then trains a Bayesian network to model the temporal evolution of quantized sparse coefficients, which captures the dependencies between feature vectors in the sequence. Signals are detected using a cumulative log-likelihood ratio test statistic for each signal type. Once a signal is detected, a similar test statistic is used to again look for a quiescent period. This allows for entire signal events to be detected and assigned a unified label, rather than potentially allowing different labels to be assigned to adjacent vectors within a signal event. For more details on the SCST method, see the paper cited above.

The Python implementation of the SCST approach provided in this repository is currently a simplified version of the full implementation just described. The following summarizes the differences between this and the full implementations:
- The sparse decomposition framework is not yet implemented, which has a notable impact on overall classification accuracy, both owing to the increased difficulty of modeling vector coefficients, and the lack of interference mitigation. This means that, depending on the data being processed, there's currently not much "sparse" about this SCST implementation.
- In this implementation, the quantizers applied to vector coefficients prior to likelihood calculation are based on the Llyod-Max formulation, whereas the original implementation uses quantizers that are based on maximizing the *J*-divergence between the quantized coefficients associated with different classes.

For benchmarking purposes, this repository also provides an implementation of a competing classifier, which extends a simple `keras`-based neural network to perform sequential classification.


## Requirements

Running this code requires Python 3 with the following packages installed:
- `numpy`
- `scipy`
- `sklearn`
- `matplotlib`
- `seaborn` (used only to plot confusion matrices)
- `keras` with a suitable backend (e.g., `tensorflow`)

If you don't already have a Python installation, it's typically easiest to install [Anaconda](https://www.anaconda.com/distribution/), since it comes with all the above packages except `seaborn` and `keras`. Installing the latter two packages can be done by typing the following commands into the Anaconda Prompt (installing `keras` should also install `tensorflow`):
- `conda install -c anaconda seaborn`
- `conda install -c anaconda keras`


## Quick Start Guide

There is no installation procedure. Simply clone the repository and set up your Python environment such that the directory associated with the cloned repository is in your `PYTHONPATH`. One way to accomplish this is to run `source` [env.sh](env.sh) in the root directory of this repository each time you start a new terminal instance.

Sample data sequences have been included in this repository, and are located in the `GRSA001` directory. This data represents sound pressure level (SPL) recordings in a natural landscape, where the objective is to detect and classify sources of man-made sounds, e.g., prop-planes and jets. Each vector has elements that represent the average sound pressure level in a different 1/3 octave frequency band, where one measurement was recorded every second.

Before the classifiers can be applied to this data, they must be trained, which can be accomplished by running
```
python train_classifiers_nvspl.py
```
This produces a `.pkl` file for each classifier, which contains a saved classifier object of the appropriate type. The trained classifiers (both SCST and `keras`) can then be applied to a default example data segment by running
```
python apply_classifiers.py
```
This will produce a confusion matrix summarizing the classification performance of each method, and an image of the data with color-coded strips above it, indicating the class label assigned by each method, as well as the true class labels. Ideally, a given classifier's ID strip would exactly match the strip associated with the true labels.

To apply the classifiers to other data segments, various command line arguments can be provided when executing the `apply_classifiers.py` script. One can supply a `truth_path` argument to apply the classifier to a predetermined segment of data, which will display the results relative to the truth, e.g.,
```
python apply_classifiers.py --truth_path GRSA001/TRUTH_GRSA001_2008_10_03_h1-3.txt
```
Alternatively, one can apply the classifiers to an arbitrary segment of data from a `NVSPL` file using the arguments `data_path`, `start_second`, and `end_second`, e.g.,
```
python apply_classifiers.py --data_path GRSA001/NVSPL_GRSA001_2008_10_03.txt --start_second 3600 --end_second 7200
```
Since no truth segment is associated with the input in this case, classification performance will not be calculated, but rather, only the ID strips for each classifier will be shown, and the user will have to visually verify the effectiveness of each approach.

Finally, one can run the unit tests included in the repository as follows
```
nosetests test
```
which requires the `nosetests` package (included with Anaconda). Scripts containing tests can be run independently without `nosetests`, if desired, e.g.,
```
python test/test_scst_model.py
```


## Notes on Performance

One of the defining characteristics of the example data in the `GRSA001` directory is that many of the time segments contain significant interference signatures, often superimposed with those of signals to be detected and classified. In the `GRSA001` data, these "interference" sources are often wind, rain/thunder, and wildlife calls (birds, elk, etc.). Superimposed interference was one of the primary motivators behind the design of the original SCST formulation. Accordingly, one can expect that neither classifier implementation included in this repository will perform well on time segments that contain significant interference signatures, as they do not yet implement interference mitigation techniques.

As they are, the overall performance of the SCST and `keras` classifiers are similar to each other on the provided data, though the SCST classifier does a better job at assigning consistent class labels to entire events, while the `keras` classifier has improved performance on some events owing to switching to the correct class label mid-event. The original MATLAB SCST classifier implementation provides a dramatic performance improvement to the implementation supplied here, as it typically handles the presence of significant interference rather gracefully. The `keras`-based classifier included in this repository could also likely be improved, as an optimal design of this classifier was not the focus of this project. Perhaps one of the best ways to improve the `keras` classifier's performance would be to adopt the SCST sparse decomposition approach as a preprocessing step to mitigate the impacts of interference. In general, one should not draw too many conclusions about the relative usefulness of the two classifiers based on the example results alone.


## Application to Other Data

Application of the `SCSTClassifier` (or `TransientKerasClassifier`) to other vector sequence data should be relatively straightforward. Each classifier is automatically trained on construction using the provided training data and parameters. See the docstring associated with each classifier's `__init__` method for an explanation of each training parameter. For more detailed guidelines on setting the SCST parameters, see the paper cited above.

The structure of the input training data is probably the most important factor to consider when constructing a new classifier. The `train_event_dict` input is a `dict` with keys that are class labels (typically `int` or `str`) with values that are `list` of training events for that class. A training event is a matrix containing the signatures of a given signal event, where feature vectors are columns of this matrix. The training data must be segmented into a `list` of events since 1) the SCST classifier forms prior distributions to help identify the start of transient events, and 2) we must ensure temporal dependencies between feature vectors are only captured within an event, and not between vectors in different events. Note that all feature vectors in every data sequence (training and testing) must have the same dimensionality. Finally, `train_event_dict` must contain a value at key `0` that is a `list` of training events for noise (or noise plus interference), i.e., the null hypothesis. A trained classifier can be saved using its `save` method, and loaded by another script using the `static` method `load`.

Each classifier can then be applied to a new test data sequence by calling the `classify` method on each feature vector in a sequence, one at a time. The SCST classifier also requires signal and noise detection thresholds, since it operates on a cumulative log-likelihood ratio test statistic, as mentioned above. The class label for a given feature vector is output by the `classify` method when it is applied to that feature vector. Alternatively, one can classify the entire sequence and reference the `class_labels` property of the classifier to obtain labels for the entire sequence.

Since both classifiers use information from previous vectors when assigning labels to a new feature vector, they must be reset to appropriately apply them to a new vector sequence; this can be accomplished using the `reset` method in each case.


## License

This project is licensed under the MIT License, the details of which can be found in the [LICENSE.md](LICENSE.md) file.
