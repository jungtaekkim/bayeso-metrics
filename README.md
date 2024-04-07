<p align="center">
<img src="https://raw.githubusercontent.com/jungtaekkim/bayeso/main/docs/_static/assets/logo_bayeso_capitalized.svg" width="400" />
</p>

# Beyond Regrets: Geometric Metrics for Bayesian Optimization
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* [https://bayeso.org](https://bayeso.org)
* [arXiv preprint](https://arxiv.org/abs/2401.01981)

## Abstract

Bayesian optimization is a principled optimization strategy for a black-box objective function. It shows its effectiveness in a wide variety of real-world applications such as scientific discovery and experimental design. In general, the performance of Bayesian optimization is reported through regret-based metrics such as instantaneous, simple, and cumulative regrets. These metrics only rely on function evaluations, so that they do not consider geometric relationships between query points and global solutions, or query points themselves. Notably, they cannot discriminate if multiple global solutions are successfully found. Moreover, they do not evaluate Bayesian optimization's abilities to exploit and explore a search space given. To tackle these issues, we propose four new geometric metrics, i.e., precision, recall, average degree, and average distance. These metrics allow us to compare Bayesian optimization algorithms considering the geometry of both query points and global optima, or query points. However, they are accompanied by an extra parameter, which needs to be carefully determined. We therefore devise the parameter-free forms of the respective metrics by integrating out the additional parameter. Finally, we empirically validate that our proposed metrics can provide more delicate interpretation of Bayesian optimization algorithms, on top of assessment via the conventional metrics.

## Available Metrics

The details of available metrics can be found in [our paper](https://arxiv.org/abs/2401.01981).

* Instantaneous regret
* Simple regret
* Cumulative regret
* Precision
* Recall
* Average degree
* Average distance
* Parameter-free precision
* Parameter-free recall
* Parameter-free average degree
* Parameter-free average distance

## Simple Usage

* Instantaneous regret, simple regret, and cumulative regret

```python
from bayeso_metrics import instantaneous_regret
from bayeso_metrics import simple_regret
from bayeso_metrics import cumulative_regret

# last_value: a function value at the last iteration
# values: function values until the last iteration
# optimum_value: a global optimum value
# num_init: the number of initial points

ireg = instantaneous_regret(last_value, optimum_value)
sreg = simple_regret(values, optimum_value, num_init)
creg = cumulative_regret(values, optimum_value, num_init)
```

* Precision, recall, average degree, and average distance

```python
from bayeso_metrics import precision
from bayeso_metrics import recall
from bayeso_metrics import average_degree
from bayeso_metrics import average_distance

# samples: acquired samples
# optima: global optima
# radius: a radius for determining nearest neighbors
# k: the number of nearest neighbors for calculating an average distance

prec, _ = precision(samples, optima, radius)
reca, _ = recall(samples, optima, radius)
degr = average_degree(samples, radius)
dist = average_distance(samples, k)
```

* Parameter-free precision, parameter-free recall, parameter-free average degree, and parameter-free average distance

```python
from bayeso_metrics import parameter_free_precision
from bayeso_metrics import parameter_free_recall
from bayeso_metrics import parameter_free_average_degree
from bayeso_metrics import parameter_free_average_distance

# samples: acquired samples
# optima: global optima
# num_samples: the number of metric samples for integrating out a radius
#              or the number of nearest neighbors (default value: 100)
# seed: a random seed (default value: None)

pfprec = parameter_free_precision(samples, optima, num_samples=num_samples, seed=seed)
pfreca = parameter_free_recall(samples, optima, num_samples=num_samples, seed=seed)
pfdegr = parameter_free_average_degree(samples, num_samples=num_samples, seed=seed)
pfdist = parameter_free_average_distance(samples, num_samples=num_samples, seed=seed)
```

## Citation
```
@article{KimJ2024arxiv,
    author={Kim, Jungtaek},
    title={Beyond Regrets: Geometric Metrics for {Bayesian} Optimization},
    journal={arXiv preprint arXiv:2401.01981},
    year={2024}
}
```

## License
[MIT License](LICENSE)
