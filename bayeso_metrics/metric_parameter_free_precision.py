import numpy as np

from bayeso_metrics import precision
from bayeso_metrics import utils_sampling


def parameter_free_precision(queries, optima, num_samples=100, seed=None):
    assert isinstance(queries, np.ndarray)
    assert isinstance(optima, np.ndarray)
    assert isinstance(num_samples, (int, np.int32, np.int64))
    assert isinstance(seed, (type(None), int, np.int32, np.int64))
    assert queries.ndim == 2
    assert optima.ndim == 2
    assert queries.shape[1] == optima.shape[1]

    dim = queries.shape[1]
    samples = utils_sampling.get_samples(num_samples, 'continuous', dim, seed=seed)
    precs = []

    for sample in samples:
        cur_prec, _ = precision(queries, optima, sample)
        precs.append(cur_prec)
    precs = np.array(precs)

    assert precs.shape[0] == samples.shape[0] == num_samples
    prec = np.mean(precs)

    return prec
