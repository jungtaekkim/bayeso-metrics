import numpy as np

from bayeso_metrics import average_degree
from bayeso_metrics import utils_sampling


def parameter_free_average_degree(queries, num_samples=100, seed=None):
    assert isinstance(queries, np.ndarray)
    assert isinstance(num_samples, (int, np.int32, np.int64))
    assert isinstance(seed, (type(None), int, np.int32, np.int64))
    assert queries.ndim == 2

    dim = queries.shape[1]
    samples = utils_sampling.get_samples(num_samples, 'continuous', dim, seed=seed)
    degrs = []

    for sample in samples:
        cur_degr = average_degree(queries, sample)
        degrs.append(cur_degr)
    degrs = np.array(degrs)

    assert degrs.shape[0] == samples.shape[0] == num_samples
    degr = np.mean(degrs)

    return degr
