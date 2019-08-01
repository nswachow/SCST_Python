import numpy as np

from scst.scst_model import SCSTElementModel, SCSTVectorModel, SCSTClassModel


def testRandomModels():

    MAX_VALUE = 5
    VECT_DIM = 4
    ORDER = 2
    MAX_NUM_DEPEND = 3
    MI_THRESH = 0.01
    MIN_PROB = 1e-4
    NUM_OBS = 100
    NUM_PRI_OBS = np.random.randint(low=10, high=NUM_OBS)
    ELEMENT_IDX = np.random.randint(low=0, high=VECT_DIM)

    train_event_list = [np.random.randint(low=0, high=MAX_VALUE, size=(VECT_DIM, NUM_OBS))
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
    NUM_OBS = 10
    NUM_PRI_OBS = 10
    ELEMENT_IDX = 1

    zero_array = np.zeros((1, NUM_OBS), dtype=int)
    alt_array = np.array([ii % 2 for ii in np.arange(NUM_OBS)]).reshape((1, NUM_OBS))
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


if __name__ == '__main__':
    testRandomModels()
    testSimpleModels()
