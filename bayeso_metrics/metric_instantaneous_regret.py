import numpy as np


def instantaneous_regret(y, global_minimum_value):
    assert isinstance(y, (float, np.float32, np.float64))
    assert isinstance(global_minimum_value, (float, np.float32, np.float64))

    value = y - global_minimum_value

    assert value >= 0.0
    return value

def instantaneous_regrets(by, global_minimum_value, num_init):
    assert isinstance(by, np.ndarray)
    assert isinstance(global_minimum_value, (float, np.float32, np.float64))
    assert isinstance(num_init, (int, np.int32, np.int64))
    assert by.ndim == 1
    assert by.shape[0] >= num_init

    values = np.array([instantaneous_regret(y, global_minimum_value) for y in by])
    values = np.concatenate([
        [np.min(values[:num_init])],
        values[num_init:]
    ], axis=0)

    assert np.all(values >= 0.0)
    assert values.shape[0] == (by.shape[0] - num_init + 1)
    return values
