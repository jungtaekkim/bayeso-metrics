def test_import_bayeso_metrics():
    import bayeso_metrics

def test_import_metrics_regrets():
    from bayeso_metrics import instantaneous_regret
    from bayeso_metrics import simple_regret
    from bayeso_metrics import cumulative_regret

    from bayeso_metrics import instantaneous_regrets
    from bayeso_metrics import simple_regrets
    from bayeso_metrics import cumulative_regrets

    assert callable(instantaneous_regret)
    assert callable(simple_regret)
    assert callable(cumulative_regret)

    assert callable(instantaneous_regrets)
    assert callable(simple_regrets)
    assert callable(cumulative_regrets)

def test_import_metrics_geometric_metrics():
    from bayeso_metrics import precision
    from bayeso_metrics import recall
    from bayeso_metrics import average_degree
    from bayeso_metrics import average_distance

    assert callable(precision)
    assert callable(recall)
    assert callable(average_degree)
    assert callable(average_distance)

def test_import_metrics_parameter_free_geometric_metrics():
    from bayeso_metrics import parameter_free_precision
    from bayeso_metrics import parameter_free_recall
    from bayeso_metrics import parameter_free_average_degree
    from bayeso_metrics import parameter_free_average_distance

    assert callable(parameter_free_precision)
    assert callable(parameter_free_recall)
    assert callable(parameter_free_average_degree)
    assert callable(parameter_free_average_distance)

def test_import_metrics_pf_geometric_metrics():
    from bayeso_metrics import pf_precision
    from bayeso_metrics import pf_recall
    from bayeso_metrics import pf_average_degree
    from bayeso_metrics import pf_average_distance

    assert callable(pf_precision)
    assert callable(pf_recall)
    assert callable(pf_average_degree)
    assert callable(pf_average_distance)

def test_import_utils():
    from bayeso_metrics import utils_metrics
    from bayeso_metrics import utils_nearest_neighbors
    from bayeso_metrics import utils_sampling
