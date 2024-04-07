import numpy as np
import matplotlib.pyplot as plt
from bayeso_benchmarks import utils as bb_utils

from bayeso_metrics import instantaneous_regret
from bayeso_metrics import simple_regret
from bayeso_metrics import cumulative_regret
from bayeso_metrics import precision
from bayeso_metrics import recall
from bayeso_metrics import average_degree
from bayeso_metrics import average_distance


def get_grids(bounds, num_grids):
    assert bounds.shape[0] == 2

    x = np.linspace(bounds[0, 0], bounds[0, 1], num_grids)
    y = np.linspace(bounds[1, 0], bounds[1, 1], num_grids)

    xx, yy = np.meshgrid(x, y)
    grids = []

    for elem in [xx, yy]:
        grids.append(elem.flatten(order='C'))

    grids = np.vstack(tuple(grids))
    grids = grids.T
    return grids, xx, yy

def plot(obj_target,
    samples=None,
    selections=None,
    radii=None,
    num_grids=100,
):
    str_target = obj_target.name
    bounds = obj_target.get_bounds()
    global_minimizers = obj_target.get_global_minimizers()
    assert bounds.shape[0] == 2
    assert global_minimizers.shape[1] == 2

    grids, xx, yy = get_grids(bounds, num_grids)
    zz = obj_target.output(grids)
    zz = np.reshape(zz, (num_grids, num_grids), order='C')

    plt.rc('text', usetex=True)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca()

    ctf = ax.contourf(xx, yy, zz, cmap='plasma', levels=np.linspace(np.min(zz), np.max(zz), 21))

    cbar = fig.colorbar(ctf)
    cbar.ax.set_ylabel(r'\textrm{Function Value}', fontsize=24)

    ax.plot(global_minimizers[:, 0], global_minimizers[:, 1],
        linestyle='none', color='red', marker='X', markersize=24,
        markeredgewidth=1, markeredgecolor='white')

    if samples is not None:
        assert samples.shape[1] == 2

        ax.plot(samples[:, 0], samples[:, 1],
            linestyle='none', color='green', marker='P', markersize=24,
            markeredgewidth=1, markeredgecolor='white')

    if samples is not None and radii is not None:
        assert samples.shape[1] == 2
        assert isinstance(radii, list)

        for sample in samples:
            for ind_radius, radius in enumerate(radii):
                assert isinstance(radius, float)
                circle = plt.Circle(tuple(sample), radius, color='yellow', alpha=0.15)
                ax.add_patch(circle)

    if selections is not None:
        assert selections.shape[1] == 2

        ax.plot(selections[:, 0], selections[:, 1],
            linestyle='none', color='blue', marker='o', markersize=12,
            markeredgewidth=1, markeredgecolor='white')

    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_xlabel(r'$x_1$', fontsize=24)
    ax.set_ylabel(r'$x_2$', fontsize=24)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    cbar.ax.tick_params(labelsize=20)

    fig.tight_layout()

    plt.show()
    plt.close('all')


if __name__ == '__main__':
    np.random.seed(42)
    str_fun = 'branin'
    num_init = 1

    list_config = [
        (10, 212, 2, 2.0),
        (10, 212, 2, 4.0),
        (10, 212, 2, 6.0),
        (10, 212, 2, 2.0),
        (10, 212, 4, 2.0),
        (10, 212, 6, 2.0),
        (20, 542, 2, 2.0),
        (20, 542, 2, 4.0),
        (20, 542, 2, 6.0),
        (20, 542, 2, 2.0),
        (20, 542, 4, 2.0),
        (20, 542, 6, 2.0),
    ]

    for num_points, seed, k, radius in list_config:
        obj = bb_utils.get_benchmark(str_fun)
        optima = obj.get_global_minimizers()
        optimum_value = obj.global_minimum

        samples = obj.sample_uniform(num_points, seed=seed)
        samples_vals = obj.output(samples)[:, 0]

        optima_vals = obj.output(optima)

        ireg = instantaneous_regret(samples_vals[-1], optimum_value)
        sreg = simple_regret(samples_vals, optimum_value, num_init)
        creg = cumulative_regret(samples_vals, optimum_value, num_init)
        prec, selections_prec = precision(samples, optima, radius)
        reca, selections_reca = recall(samples, optima, radius)
        degr = average_degree(samples, radius)
        dist = average_distance(samples, k)

        print(f'num_points {num_points} seed {seed} k {k} radius {radius}')
        print(f'instantaneous regret {ireg:.4f}')
        print(f'simple regret {sreg:.4f}')
        print(f'cumulative regret {creg:.4f}')
        print(f'precision {prec:.4f}')
        print(f'recall {reca:.4f}')
        print(f'average_degree {degr:.4f}')
        print(f'average_distance {dist:.4f}')
        print('')

        plot(obj, samples=samples, selections=selections_prec, radii=[radius])
        plot(obj, samples=samples, selections=selections_reca, radii=[radius])
