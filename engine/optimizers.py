"""Optimizers update parameters in place from their gradients."""
import numpy as np


class SGD:
    name = "sgd"

    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * g


class Adam:
    name = "adam"

    def __init__(self, lr: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = None
        self.v = None

    def step(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def get_optimizer(name: str, lr: float):
    return {"sgd": SGD, "adam": Adam}[name](lr=lr)
