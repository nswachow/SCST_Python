import numpy as np

from scst.quantizer import LloydMaxQuantizer


def testLloydMaxQuantizer():

    LOW = 0
    HIGH = 10
    NUM_LEVELS = 4

    value_array = np.random.uniform(low=LOW, high=HIGH, size=100)
    cont_quantizer = LloydMaxQuantizer(NUM_LEVELS, value_array, False)
    discrete_quantizer = LloydMaxQuantizer(NUM_LEVELS, value_array, True)

    # Ensure reconstruction values are as expected for a continuous quantizer
    transition_levels = cont_quantizer.transition_levels
    reconstruction_levels = cont_quantizer.reconstruction_levels
    assert np.isclose(cont_quantizer(transition_levels[0] - 1), reconstruction_levels[0])
    assert np.isclose(cont_quantizer(transition_levels[-1] + 1), reconstruction_levels[-1])

    for ii in range(len(transition_levels)-1):
        eval_value = (transition_levels[ii] + transition_levels[ii]) / 2
        assert np.isclose(cont_quantizer(eval_value), reconstruction_levels[ii+1])

    # Ensure reconstruction values are as expected for a discrete quantizer
    transition_levels = discrete_quantizer.transition_levels
    reconstruction_levels = discrete_quantizer.reconstruction_levels
    assert np.isclose(discrete_quantizer(transition_levels[0] - 1), 0)
    assert np.isclose(discrete_quantizer(transition_levels[-1] + 1), NUM_LEVELS-1)

    for ii in range(len(transition_levels)-1):
        eval_value = (transition_levels[ii] + transition_levels[ii]) / 2
        assert np.isclose(discrete_quantizer(eval_value), ii+1)


if __name__ == '__main__':
    testLloydMaxQuantizer()
