"""2D toy datasets so the decision boundary can be drawn."""
import numpy as np


def _moons(n, noise, rng):
    t = rng.uniform(0, np.pi, n // 2)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.vstack([a, b]) + rng.normal(0, noise, (2 * (n // 2), 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y


def _circles(n, noise, rng):
    t = rng.uniform(0, 2 * np.pi, n // 2)
    outer = np.c_[np.cos(t), np.sin(t)]
    inner = 0.45 * np.c_[np.cos(t), np.sin(t)]
    X = np.vstack([outer, inner]) + rng.normal(0, noise, (2 * (n // 2), 2))
    y = np.r_[np.zeros(n // 2), np.ones(n // 2)]
    return X, y


def _spiral(n, noise, rng):
    m = n // 2
    r = np.linspace(0.1, 1.0, m)
    t = np.linspace(0, 3.2 * np.pi, m)
    a = np.c_[r * np.cos(t), r * np.sin(t)]
    b = np.c_[r * np.cos(t + np.pi), r * np.sin(t + np.pi)]
    X = np.vstack([a, b]) + rng.normal(0, noise, (2 * m, 2))
    y = np.r_[np.zeros(m), np.ones(m)]
    return X, y


def _xor(n, noise, rng):
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    X = X + rng.normal(0, noise * 0.3, X.shape)
    return X, y


_DATASETS = {"moons": _moons, "circles": _circles, "spiral": _spiral, "xor": _xor}


def make_dataset(name: str = "moons", n: int = 200, noise: float = 0.12, seed: int = 0):
    """Returns X with shape (n, 2) and y with shape (n, 1), labels 0 or 1."""
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(_DATASETS)}")
    rng = np.random.default_rng(seed)
    X, y = _DATASETS[name](n, noise, rng)
    return X.astype(float), y.reshape(-1, 1).astype(float)
