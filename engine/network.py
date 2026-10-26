"""A small feed-forward network built only with NumPy."""
import numpy as np

from .activations import get_activation
from .layers import Dense
from .losses import get_loss
from .optimizers import get_optimizer


class Network:
    def __init__(
        self,
        sizes,
        activation: str = "tanh",
        loss: str = "bce",
        optimizer: str = "adam",
        lr: float = 0.03,
        seed: int = 42,
    ):
        """sizes example: [2, 8, 8, 1] = 2 inputs, two hidden layers of 8, 1 output."""
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.layers = []
        for i in range(len(sizes) - 1):
            is_last = i == len(sizes) - 2
            act = get_activation("sigmoid" if is_last else activation)
            self.layers.append(Dense(sizes[i], sizes[i + 1], act, rng))
        self.loss = get_loss(loss)
        self.optimizer = get_optimizer(optimizer, lr)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_step(self, X: np.ndarray, y: np.ndarray):
        """One full-batch gradient step. Returns (loss, accuracy)."""
        pred = self.forward(X)
        loss = self.loss.forward(pred, y)
        self.backward(self.loss.backward(pred, y))
        params = [p for layer in self.layers for p in layer.params]
        grads = [g for layer in self.layers for g in layer.grads]
        self.optimizer.step(params, grads)
        acc = float(np.mean((pred > 0.5) == (y > 0.5)))
        return loss, acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def weights(self):
        """Weight matrices per layer, shape [layer][in][out]. Used by the UI graph."""
        return [layer.W.tolist() for layer in self.layers]

    def n_params(self) -> int:
        return sum(p.size for layer in self.layers for p in layer.params)
