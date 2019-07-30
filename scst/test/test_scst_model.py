import numpy as np
from sys import float_info

from scst.scst_model import SCSTElementModel


def testRandomElementModel():

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

    model = SCSTElementModel(train_event_list, ELEMENT_IDX, ORDER, MAX_NUM_DEPEND, MI_THRESH,
                             MIN_PROB, MAX_VALUE, NUM_PRI_OBS)

    for _ in range(10):
        data_matrix = np.random.random((VECT_DIM, ORDER+1))
        model.logLikelihood(data_matrix)


def testSimpleElementModel():

    MAX_VALUE = 1
    ORDER = 1
    MAX_NUM_DEPEND = 1
    MI_THRESH = 0.01
    MIN_PROB = 0
    NUM_OBS = 10
    NUM_PRI_OBS = None
    ELEMENT_IDX = 1

    zero_array = np.zeros((1, NUM_OBS), dtype=int)
    alt_array = np.array([ii % 2 for ii in np.arange(NUM_OBS)]).reshape((1, NUM_OBS))
    train_event_list = [np.concatenate((zero_array, alt_array))]

    model = SCSTElementModel(train_event_list, ELEMENT_IDX, ORDER, MAX_NUM_DEPEND, MI_THRESH,
                             MIN_PROB, MAX_VALUE, NUM_PRI_OBS)

    data_matrix = np.array([[0, 0], [0, 1]])
    ll = model.logLikelihood(data_matrix)
    assert np.isclose(ll, 0.0)

    data_matrix = np.array([[0, 0], [1, 1]])
    ll = model.logLikelihood(data_matrix)
    assert np.isclose(ll, float_info.min)


if __name__ == '__main__':
    testRandomElementModel()
    testSimpleElementModel()
