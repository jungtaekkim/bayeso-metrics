import numpy as np
from sklearn.neighbors import NearestNeighbors


def k_nn(candidates, queries, num_nearest_neighbors):
    assert isinstance(candidates, np.ndarray)
    assert isinstance(queries, np.ndarray)
    assert isinstance(num_nearest_neighbors, (int, np.int64))
    assert len(candidates.shape) == 2
    assert len(queries.shape) == 2
    assert candidates.shape[1] == queries.shape[1]
    assert num_nearest_neighbors > 0

    model = NearestNeighbors(n_neighbors=num_nearest_neighbors + 1, algorithm='kd_tree') # 1 is added because nearest neighbors always contain a query itself.
    model.fit(candidates)

    distances, indices = model.kneighbors(queries, return_distance=True)
    nearest_neighbors = candidates[indices]

    return nearest_neighbors, distances

def radius_nn(candidates, queries, radius):
    assert isinstance(candidates, np.ndarray)
    assert isinstance(queries, np.ndarray)
    assert isinstance(radius, float)
    assert len(candidates.shape) == 2
    assert len(queries.shape) == 2
    assert candidates.shape[1] == queries.shape[1]
    assert radius > 0.0

    model = NearestNeighbors(radius=radius, algorithm='kd_tree')
    model.fit(candidates)

    _, indices = model.radius_neighbors(queries, return_distance=True, sort_results=False)
    nearest_neighbors = []

    for indices_query in indices:
        nearest_neighbors.append(candidates[indices_query])

    return nearest_neighbors
