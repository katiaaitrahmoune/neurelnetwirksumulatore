"""Activation functions. Each one has a forward `fn` and a `grad` used in backprop."""
import numpy as np


class Activation:
    name = "base"

    def fn(self, z):
        raise NotImplementedError

    def grad(self, z, a):
        """Derivative of the activation, given the input z and the output a."""
        raise NotImplementedError


class ReLU(Activation):
    name = "relu"

    def fn(self, z):
        return np.maximum(0.0, z)

    def grad(self, z, a):
        return (z > 0).astype(z.dtype)


class Sigmoid(Activation):
    name = "sigmoid"

    def fn(self, z):
        z = np.clip(z, -60, 60)
        return 1.0 / (1.0 + np.exp(-z))

    def grad(self, z, a):
        return a * (1.0 - a)


class Tanh(Activation):
    name = "tanh"

    def fn(self, z):
        return np.tanh(z)

    def grad(self, z, a):
        return 1.0 - a * a


class Linear(Activation):
    name = "linear"

    def fn(self, z):
        return z

    def grad(self, z, a):
        return np.ones_like(z)


_REGISTRY = {c.name: c for c in (ReLU, Sigmoid, Tanh, Linear)}


def get_activation(name: str) -> Activation:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(_REGISTRY)}")
    return _REGISTRY[name]()
