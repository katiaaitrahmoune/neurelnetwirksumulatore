"""Loss functions. `forward` returns a scalar, `backward` returns dLoss/d(prediction)."""
import numpy as np

EPS = 1e-9


class MSE:
    name = "mse"

    def forward(self, pred, target):
        return float(np.mean((pred - target) ** 2))

    def backward(self, pred, target):
        return 2.0 * (pred - target) / pred.size


class BinaryCrossEntropy:
    name = "bce"

    def forward(self, pred, target):
        p = np.clip(pred, EPS, 1.0 - EPS)
        return float(-np.mean(target * np.log(p) + (1.0 - target) * np.log(1.0 - p)))

    def backward(self, pred, target):
        p = np.clip(pred, EPS, 1.0 - EPS)
        return (p - target) / (p * (1.0 - p)) / pred.size


def get_loss(name: str):
    return {"mse": MSE, "bce": BinaryCrossEntropy}[name]()
