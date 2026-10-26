"""Fully connected layer with manual backpropagation."""
import numpy as np

from .activations import Activation


class Dense:
    def __init__(self, n_in: int, n_out: int, activation: Activation, rng: np.random.Generator):
        # Xavier for tanh/sigmoid/linear, He for relu
        scale = np.sqrt(2.0 / n_in) if activation.name == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.normal(0.0, scale, size=(n_in, n_out))
        self.b = np.zeros((1, n_out))
        self.activation = activation
        self._x = None
        self._z = None
        self._a = None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z = x @ self.W + self.b
        self._a = self.activation.fn(self._z)
        return self._a

    def backward(self, grad_a: np.ndarray) -> np.ndarray:
        """grad_a is dLoss/d(output). Returns dLoss/d(input)."""
        dz = grad_a * self.activation.grad(self._z, self._a)
        self.dW = self._x.T @ dz
        self.db = dz.sum(axis=0, keepdims=True)
        return dz @ self.W.T

    @property
    def params(self):
        return [self.W, self.b]

    @property
    def grads(self):
        return [self.dW, self.db]
