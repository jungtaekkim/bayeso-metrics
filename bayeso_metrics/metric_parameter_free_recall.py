import numpy as np

from bayeso_metrics import recall
from bayeso_metrics import utils_sampling


def parameter_free_recall(queries, optima, num_samples=100, seed=None):
    assert isinstance(queries, np.ndarray)
    assert isinstance(optima, np.ndarray)
    assert isinstance(num_samples, (int, np.int32, np.int64))
    assert isinstance(seed, (type(None), int, np.int32, np.int64))
    assert queries.ndim == 2
    assert optima.ndim == 2
    assert queries.shape[1] == optima.shape[1]

    dim = queries.shape[1]
    samples = utils_sampling.get_samples(num_samples, 'continuous', dim, seed=seed)
    recas = []

    for sample in samples:
        cur_reca, _ = recall(queries, optima, sample)
        recas.append(cur_reca)
    recas = np.array(recas)

    assert recas.shape[0] == samples.shape[0] == num_samples
    reca = np.mean(recas)

    return reca
