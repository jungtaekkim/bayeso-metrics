import numpy as np

from bayeso_metrics import utils_nearest_neighbors


def average_degree(queries, radius):
    assert isinstance(queries, np.ndarray)
    assert isinstance(radius, (float, np.float32, np.float64))
    assert queries.ndim == 2
    assert radius >= 0.0

    count = 0.0

    nearest_neighbors = utils_nearest_neighbors.radius_nn(
        queries,
        queries,
        radius,
    )
    assert len(nearest_neighbors) == queries.shape[0]

    for nearest_neighbors_query in nearest_neighbors:
        count += nearest_neighbors_query.shape[0] - 1

    degr = count / queries.shape[0]
    return degr
