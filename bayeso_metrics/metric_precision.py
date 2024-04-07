import numpy as np

from bayeso_metrics import utils_nearest_neighbors


def precision(queries, optima, radius):
    assert isinstance(queries, np.ndarray)
    assert isinstance(optima, np.ndarray)
    assert isinstance(radius, (float, np.float32, np.float64))
    assert queries.ndim == 2
    assert optima.ndim == 2
    assert queries.shape[1] == optima.shape[1]
    assert radius >= 0.0

    count = 0.0
    selections = np.zeros((0, optima.shape[1]))

    nearest_neighbors = utils_nearest_neighbors.radius_nn(
        optima,
        queries,
        radius,
    )

    for nearest_neighbors_query, query in zip(nearest_neighbors, queries):
        existence = 0.0

        for optimum in optima:
            comparisons = np.all(optimum == nearest_neighbors_query, axis=1)
            assert np.sum(comparisons) == 0.0 or np.sum(comparisons) == 1.0

            if np.sum(comparisons) == 1.0:
                existence = 1.0
                selections = np.concatenate((selections, [query]), axis=0)

        assert existence == 0.0 or existence == 1.0
        count += existence

    prec = count / queries.shape[0]
    assert 0.0 <= prec and prec <= 1.0

    selections = np.unique(selections, axis=0)
    assert selections.shape[0] <= queries.shape[0]
    assert selections.shape[1] == optima.shape[1]

    return prec, selections
