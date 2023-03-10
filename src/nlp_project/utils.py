import numpy as np
from sklearn.preprocessing import normalize


def softmax(zs: np.ndarray):
    exps = np.exp(zs - np.max(zs))
    return exps / np.sum(exps, axis=-1, keepdims=True)


def kldivergence(zs: np.ndarray):
    unif = np.ones(zs.shape[1])
    return np.sum(np.multiply(np.log(np.divide(zs, unif)), zs), axis=1)


def sampled_sphere(n_dirs: int, d: int):
    """Produce ndirs samples of d-dimensional uniform distribution on the
    unit sphere
    """

    mean = np.zeros(d)
    identity = np.identity(d)
    U = np.random.multivariate_normal(mean=mean, cov=identity, size=n_dirs)

    return normalize(U)
