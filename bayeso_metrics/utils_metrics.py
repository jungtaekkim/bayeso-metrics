from bayeso_metrics import instantaneous_regret
from bayeso_metrics import simple_regret
from bayeso_metrics import cumulative_regret

from bayeso_metrics import precision
from bayeso_metrics import recall
from bayeso_metrics import average_degree
from bayeso_metrics import average_distance

from bayeso_metrics import parameter_free_precision
from bayeso_metrics import parameter_free_recall
from bayeso_metrics import parameter_free_average_degree
from bayeso_metrics import parameter_free_average_distance


def calculate_regrets(by, global_minimum_value, num_init):
    ir = instantaneous_regret(by[-1], global_minimum_value)
    sr = simple_regret(by, global_minimum_value, num_init)
    cr = cumulative_regret(by, global_minimum_value, num_init)

    return {
        'instantaneous_regret': ir,
        'simple_regret': sr,
        'cumulative_regret': cr
    }

def calculate_geometric_metrics(X, global_minima, radius, num_nearest_neighbors):
    pr, _ = precision(X, global_minima, radius)
    re, _ = recall(X, global_minima, radius)
    de = average_degree(X, radius)
    di = average_distance(X, num_nearest_neighbors)

    return {
        'precision': pr,
        'recall': re,
        'average_degree': de,
        'average_distance': di
    }

def calculate_parameter_free_geometric_metrics(X, global_minima, num_samples, seed):
    pr = parameter_free_precision(X, global_minima, num_samples=num_samples, seed=seed)
    re = parameter_free_recall(X, global_minima, num_samples=num_samples, seed=seed)
    de = parameter_free_average_degree(X, num_samples=num_samples, seed=seed)
    di = parameter_free_average_distance(X, num_samples=num_samples, seed=seed)

    return {
        'parameter_free_precision': pr,
        'parameter_free_recall': re,
        'parameter_free_average_degree': de,
        'parameter_free_average_distance': di
    }
