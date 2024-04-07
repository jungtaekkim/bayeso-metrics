import numpy as np
import argparse

from bayeso import bo
from bayeso_benchmarks import utils as bb_utils
from bayeso_metrics import utils_metrics


def get_output(fun_target, bx):
    output = fun_target(bx)
    output = output[0, 0]

    assert isinstance(output, float)
    return output

def run_bo(obj_target, str_acquisition, num_init, num_iter, seed):
    bounds = obj_target.get_bounds()

    model_bo = bo.BOwGP(bounds,
        str_cov='matern52', str_acq=str_acquisition, normalize_Y=True,
        str_optimizer_method_bo='L-BFGS-B',
    )

    X = model_bo.get_initials('uniform', num_init, seed=seed)
    by = np.zeros((0, ))

    for bx in X:
        y = get_output(obj_target.output, bx)
        by = np.concatenate([by, [y]], axis=0)

    for ind_iter in range(0, num_iter):
        next_point, _ = model_bo.optimize(X, np.array(by)[..., np.newaxis], seed=seed + 101 * (ind_iter + 1))
        y = get_output(obj_target.output, next_point)

        X = np.concatenate([X, [next_point]], axis=0)
        by = np.concatenate([by, [y]], axis=0)

        radius = np.linalg.norm(bounds[:, 1] - bounds[:, 0], ord=2) / 10.0
        k = np.minimum(np.maximum(X.shape[0] // 5, 1), 5)
        num_samples = 100

        regrets = utils_metrics.calculate_regrets(by, obj_target.global_minimum, num_init)
        geometric_metrics = utils_metrics.calculate_geometric_metrics(X, obj_target.global_minimizers, radius, k)
        pf_geometric_metrics = utils_metrics.calculate_parameter_free_geometric_metrics(X, obj_target.global_minimizers, num_samples, seed)

        print(f'Iteration {ind_iter+1:04d}')
        print(f'Instantaneous Regret {regrets["instantaneous_regret"]:.4f}')
        print(f'Simple Regret {regrets["simple_regret"]:.4f}')
        print(f'Cumulative Regret {regrets["cumulative_regret"]:.4f}')
        print(f'Precision {geometric_metrics["precision"]:.4f}')
        print(f'Recall {geometric_metrics["recall"]:.4f}')
        print(f'Average Degree {geometric_metrics["average_degree"]:.4f}')
        print(f'Average Distance {geometric_metrics["average_distance"]:.4f}')
        print(f'Parameter-Free Precision {pf_geometric_metrics["parameter_free_precision"]:.4f}')
        print(f'Parameter-Free Recall {pf_geometric_metrics["parameter_free_recall"]:.4f}')
        print(f'Parameter-Free Average Degree {pf_geometric_metrics["parameter_free_average_degree"]:.4f}')
        print(f'Parameter-Free Average Distance {pf_geometric_metrics["parameter_free_average_distance"]:.4f}')
        print('')

    assert X.shape[0] == by.shape[0]

    radius = np.linalg.norm(bounds[:, 1] - bounds[:, 0], ord=2) / 10.0
    k = np.minimum(np.maximum(X.shape[0] // 5, 1), 5)
    num_samples = 100

    regrets = utils_metrics.calculate_regrets(by, obj_target.global_minimum, num_init)
    geometric_metrics = utils_metrics.calculate_geometric_metrics(X, obj_target.global_minimizers, radius, k)
    pf_geometric_metrics = utils_metrics.calculate_parameter_free_geometric_metrics(X, obj_target.global_minimizers, num_samples, seed)

    print('Final Results')
    print(f'Instantaneous Regret {regrets["instantaneous_regret"]:.4f}')
    print(f'Simple Regret {regrets["simple_regret"]:.4f}')
    print(f'Cumulative Regret {regrets["cumulative_regret"]:.4f}')
    print(f'Precision {geometric_metrics["precision"]:.4f}')
    print(f'Recall {geometric_metrics["recall"]:.4f}')
    print(f'Average Degree {geometric_metrics["average_degree"]:.4f}')
    print(f'Average Distance {geometric_metrics["average_distance"]:.4f}')
    print(f'Parameter-Free Precision {pf_geometric_metrics["parameter_free_precision"]:.4f}')
    print(f'Parameter-Free Recall {pf_geometric_metrics["parameter_free_recall"]:.4f}')
    print(f'Parameter-Free Average Degree {pf_geometric_metrics["parameter_free_average_degree"]:.4f}')
    print(f'Parameter-Free Average Distance {pf_geometric_metrics["parameter_free_average_distance"]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--acquisition', type=str, required=True)
    parser.add_argument('--num_init', type=int, required=True)
    parser.add_argument('--num_iter', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)

    args = parser.parse_args()

    str_target = args.target
    str_acquisition = args.acquisition
    num_init = args.num_init
    num_iter = args.num_iter
    seed = args.seed

    list_str_target = str_target.split('_')
    if len(list_str_target) == 1:
        obj_target = bb_utils.get_benchmark(list_str_target[0])
    elif len(list_str_target) == 2:
        obj_target = bb_utils.get_benchmark(list_str_target[0], dim=int(list_str_target[1]))
    else:
        raise ValueError

    run_bo(obj_target, str_acquisition, num_init, num_iter, seed)
