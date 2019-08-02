import numpy as np

from scst.scst_model import SCSTElementModel, SCSTVectorModel, SCSTClassModel, SCSTClassifier
from scst.quantizer import LloydMaxQuantizer


def testRandomModels():

    MAX_VALUE = 5
    ORDER = 2
    MAX_NUM_DEPEND = 3
    MI_THRESH = 0.01
    MIN_PROB = 1e-4

    OBS_PER_EVENT = 100
    VECT_DIM = 4
    NUM_PRI_OBS = np.random.randint(low=10, high=OBS_PER_EVENT)
    ELEMENT_IDX = np.random.randint(low=0, high=VECT_DIM)

    train_event_list = [np.random.randint(low=0, high=MAX_VALUE, size=(VECT_DIM, OBS_PER_EVENT))
                        for _ in range(10)]

    element_model = SCSTElementModel(train_event_list, ELEMENT_IDX, ORDER, MAX_NUM_DEPEND,
                                     MI_THRESH, MIN_PROB, MAX_VALUE)
    vector_model = SCSTVectorModel(train_event_list, ORDER, MAX_NUM_DEPEND, MI_THRESH, MIN_PROB,
                                   MAX_VALUE)
    class_model = SCSTClassModel(train_event_list, MAX_NUM_DEPEND, MI_THRESH, MIN_PROB, MAX_VALUE,
                                 False, NUM_PRI_OBS)

    for _ in range(10):
        data_matrix = np.random.random((VECT_DIM, ORDER+1))
        element_model.logLikelihood(data_matrix)
        vector_model.logLikelihood(data_matrix)

        for col_idx in range(data_matrix.shape[1]):
            class_model.logLikelihood(data_matrix[:, col_idx])


def testSimpleModels():

    MAX_VALUE = 1
    ORDER = 1
    MAX_NUM_DEPEND = 1
    MI_THRESH = 0.01
    MIN_PROB = 1e-4
    OBS_PER_EVENT = 10
    NUM_PRI_OBS = 10
    ELEMENT_IDX = 1

    zero_array = np.zeros((1, OBS_PER_EVENT), dtype=int)
    alt_array = np.array([ii % 2 for ii in np.arange(OBS_PER_EVENT)]).reshape((1, OBS_PER_EVENT))
    train_event_list = [np.concatenate((zero_array, alt_array))]

    element_model = SCSTElementModel(train_event_list, ELEMENT_IDX, ORDER, MAX_NUM_DEPEND,
                                     MI_THRESH, MIN_PROB, MAX_VALUE)
    vector_model = SCSTVectorModel(train_event_list, ORDER, MAX_NUM_DEPEND, MI_THRESH, MIN_PROB,
                                   MAX_VALUE)
    class_model = SCSTClassModel(train_event_list, MAX_NUM_DEPEND, MI_THRESH, MIN_PROB, MAX_VALUE,
                                 False, NUM_PRI_OBS)

    data_matrix = np.array([[0, 0], [0, 1]])
    log_likelihood = element_model.logLikelihood(data_matrix)
    assert np.isclose(log_likelihood, np.log(1-MIN_PROB))

    log_likelihood = vector_model.logLikelihood(data_matrix)
    assert np.isclose(log_likelihood, 2 * np.log(1-MIN_PROB))

    for col_idx in range(data_matrix.shape[1]):
        log_likelihood = class_model.logLikelihood(data_matrix[:, col_idx])
    assert np.isclose(log_likelihood, 2 * np.log(1-MIN_PROB))

    data_matrix = np.array([[0, 1], [1, 1]])
    log_likelihood = element_model.logLikelihood(data_matrix)
    assert np.isclose(log_likelihood, np.log(MIN_PROB))

    log_likelihood = vector_model.logLikelihood(data_matrix)
    assert np.isclose(log_likelihood, 2 * np.log(MIN_PROB))

    for col_idx in range(data_matrix.shape[1]):
        log_likelihood = class_model.logLikelihood(data_matrix[:, col_idx])
    assert np.isclose(log_likelihood, 2 * np.log(MIN_PROB))


def testSCSTClassifierRandom():

    QUANTIZER_TYPE = LloydMaxQuantizer
    NUM_QUANTIZE_LEVELS = 5
    MAX_NUM_DEPEND = 5
    MI_THRESH = 0.01
    MIN_PROB = 1e-4
    NUM_PRI_OBS = 10

    A_LOW = -3
    A_HIGH = 0.5
    B_LOW = -0.5
    B_HIGH = 0.5
    NOISE_LOW = -0.5
    NOISE_HIGH = 3

    VECT_DIM = 4
    OBS_PER_EVENT = 100

    SIG_THRESH = 10
    NOISE_THRESH = 10

    train_event_dict = {
        'A': [np.random.uniform(low=A_LOW, high=A_HIGH, size=(VECT_DIM, OBS_PER_EVENT))
              for _ in range(10)],
        'B': [np.random.uniform(low=B_LOW, high=B_HIGH, size=(VECT_DIM, OBS_PER_EVENT))
              for _ in range(10)]
    }

    noise_train_array = np.random.uniform(low=NOISE_LOW, high=NOISE_HIGH,
                                          size=((VECT_DIM, OBS_PER_EVENT)))

    classifier = SCSTClassifier(train_event_dict, noise_train_array, QUANTIZER_TYPE,
                                NUM_QUANTIZE_LEVELS, MAX_NUM_DEPEND, MI_THRESH, MIN_PROB,
                                NUM_PRI_OBS)

    test_array = np.concatenate((
        np.random.uniform(low=NOISE_LOW, high=NOISE_HIGH, size=((VECT_DIM, 20))),
        np.random.uniform(low=A_LOW, high=A_HIGH, size=((VECT_DIM, 50))),
        np.random.uniform(low=NOISE_LOW, high=NOISE_HIGH, size=((VECT_DIM, 40))),
        np.random.uniform(low=B_LOW, high=B_HIGH, size=((VECT_DIM, 70))),
        np.random.uniform(low=NOISE_LOW, high=NOISE_HIGH, size=((VECT_DIM, 30)))),
        axis=1)

    for test_idx in range(test_array.shape[1]):
        classifier.classify(test_array[:, test_idx], SIG_THRESH, NOISE_THRESH)

    num_vect = test_array.shape[1]
    assert len(classifier.class_labels) == num_vect
    assert len(classifier.noise_likelihoods) == num_vect

    for label in ['A', 'B']:
        assert len(classifier.signal_llrs[label]) == num_vect
        assert len(classifier.signal_updates[label]) == num_vect
        assert len(classifier.signal_likelihoods[label]) == num_vect
        assert len(classifier.noise_llrs[label]) == num_vect
        assert len(classifier.noise_updates[label]) == num_vect

    classifier.reset()
    assert not classifier.class_labels
    assert not classifier.signal_llrs
    assert not classifier.signal_updates
    assert not classifier.signal_likelihoods
    assert not classifier.noise_llrs
    assert not classifier.noise_updates
    assert not classifier.noise_likelihoods


if __name__ == '__main__':
    testRandomModels()
    testSimpleModels()
    testSCSTClassifierRandom()
