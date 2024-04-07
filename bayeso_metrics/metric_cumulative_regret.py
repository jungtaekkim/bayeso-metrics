import numpy as np

from bayeso_metrics import instantaneous_regrets


def cumulative_regret(by, global_minimum_value, num_init):
    assert isinstance(by, np.ndarray)
    assert isinstance(global_minimum_value, (float, np.float32, np.float64))
    assert isinstance(num_init, (int, np.int32, np.int64))
    assert by.ndim == 1
    assert by.shape[0] >= num_init

    values = cumulative_regrets(by, global_minimum_value, num_init)
    value = values[-1]

    assert value >= 0.0
    return value

def cumulative_regrets(by, global_minimum_value, num_init):
    assert isinstance(by, np.ndarray)
    assert isinstance(global_minimum_value, (float, np.float32, np.float64))
    assert isinstance(num_init, (int, np.int32, np.int64))
    assert by.ndim == 1
    assert by.shape[0] >= num_init

    values = instantaneous_regrets(by, global_minimum_value, num_init)
    values = np.cumsum(values)

    assert np.all(values >= 0.0)
    assert values.shape[0] == (by.shape[0] - num_init + 1)
    return values
