import numpy as np

from engine import Network, make_dataset


def test_learns_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)
    net = Network([2, 6, 1], activation="tanh", optimizer="adam", lr=0.05)
    for _ in range(1500):
        loss, acc = net.train_step(X, y)
    assert acc == 1.0
    assert loss < 0.1


def test_gradient_check():
    """Backprop gradients must match numerical gradients."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2))
    y = (rng.random((5, 1)) > 0.5).astype(float)
    net = Network([2, 4, 1], activation="tanh", optimizer="sgd", lr=0.0)
    pred = net.forward(X)
    net.backward(net.loss.backward(pred, y))
    W = net.layers[0].W
    analytic = net.layers[0].dW[0, 0]
    eps = 1e-5
    W[0, 0] += eps
    plus = net.loss.forward(net.forward(X), y)
    W[0, 0] -= 2 * eps
    minus = net.loss.forward(net.forward(X), y)
    W[0, 0] += eps
    numeric = (plus - minus) / (2 * eps)
    assert abs(analytic - numeric) < 1e-6


def test_moons_accuracy():
    X, y = make_dataset("moons", n=200)
    net = Network([2, 8, 8, 1], activation="tanh", optimizer="adam", lr=0.03)
    for _ in range(800):
        loss, acc = net.train_step(X, y)
    assert acc > 0.93
