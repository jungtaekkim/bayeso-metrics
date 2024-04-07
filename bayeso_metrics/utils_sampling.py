import numpy as np


def get_samples(num_samples, str_continuous_or_discrete, dim, seed=None):
    assert isinstance(num_samples, (int, np.int32, np.int64))
    assert isinstance(str_continuous_or_discrete, str)
    assert isinstance(dim, int)
    assert isinstance(seed, (type(None), int, np.int32, np.int64))
    assert num_samples > 0
    assert str_continuous_or_discrete in ['continuous', 'discrete']

    random_state = np.random.RandomState(seed)

    if str_continuous_or_discrete == 'continuous':
        scale = 0.05 * dim

        samples = random_state.exponential(scale=scale, size=(num_samples, ))
    elif str_continuous_or_discrete == 'discrete':
        p = 0.5

        samples = random_state.geometric(p=p, size=(num_samples, ))

    return samples
