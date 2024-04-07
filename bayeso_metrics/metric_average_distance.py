import numpy as np

from bayeso_metrics import utils_nearest_neighbors


def average_distance(queries, num_nearest_neighbors):
    assert isinstance(queries, np.ndarray)
    assert isinstance(num_nearest_neighbors, (int, np.int32, np.int64))
    assert queries.ndim == 2
    assert num_nearest_neighbors > 0
    assert queries.shape[0] > num_nearest_neighbors

    total_distance = 0.0

    _, distances = utils_nearest_neighbors.k_nn(
        queries,
        queries,
        num_nearest_neighbors,
    )
    assert distances.shape[0] == queries.shape[0]

    for distances_query in distances:
        assert num_nearest_neighbors == (distances_query.shape[0] - 1)
        assert distances_query[0] == 0.0

        distances_query = distances_query[1:]
        total_distance += np.sum(distances_query)

    dist = total_distance / (distances.shape[0] * num_nearest_neighbors)
    return dist
