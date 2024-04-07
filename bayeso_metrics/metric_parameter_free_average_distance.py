import numpy as np

from bayeso_metrics import average_distance
from bayeso_metrics import utils_sampling


def parameter_free_average_distance(queries, num_samples=100, seed=None):
    assert isinstance(queries, np.ndarray)
    assert isinstance(num_samples, (int, np.int32, np.int64))
    assert isinstance(seed, (type(None), int, np.int32, np.int64))
    assert queries.ndim == 2

    dim = queries.shape[1]
    samples = utils_sampling.get_samples(num_samples, 'discrete', dim, seed=seed)
    dists = []

    for sample in samples:
        if sample >= queries.shape[0]:
            sample = queries.shape[0] - 1
        cur_dist = average_distance(queries, sample)
        dists.append(cur_dist)
    dists = np.array(dists)

    assert dists.shape[0] == samples.shape[0] == num_samples
    dist = np.mean(dists)

    return dist
