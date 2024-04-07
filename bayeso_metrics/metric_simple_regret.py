import numpy as np

from bayeso_metrics import instantaneous_regrets


def simple_regret(by, global_minimum_value, num_init):
    assert isinstance(by, np.ndarray)
    assert isinstance(global_minimum_value, (float, np.float32, np.float64))
    assert isinstance(num_init, (int, np.int32, np.int64))
    assert by.ndim == 1
    assert by.shape[0] >= num_init

    values = simple_regrets(by, global_minimum_value, num_init)
    value = values[-1]

    assert value >= 0.0
    return value

def simple_regrets(by, global_minimum_value, num_init):
    assert isinstance(by, np.ndarray)
    assert isinstance(global_minimum_value, (float, np.float32, np.float64))
    assert isinstance(num_init, (int, np.int32, np.int64))
    assert by.ndim == 1
    assert by.shape[0] >= num_init

    values = instantaneous_regrets(by, global_minimum_value, num_init)
    new_values = []

    for ind in range(0, values.shape[0]):
        new_values.append(np.min(values[:ind + 1]))
    values = np.array(new_values)

    assert np.all(values >= 0.0)
    assert values.shape[0] == (by.shape[0] - num_init + 1)
    return values
