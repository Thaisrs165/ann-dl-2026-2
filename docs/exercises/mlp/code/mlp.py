"""A from-scratch multi-layer perceptron, shared by Exercises 2, 3 and 4.

Everything a network needs to learn is written here in plain ``numpy``:
the activations, the two losses, the forward pass, the backward pass
(every gradient derived by hand, no autodiff) and the gradient-descent
update. No TensorFlow, PyTorch or ``MLPClassifier`` is used anywhere.

The network is configured entirely by arguments:

* ``layer_sizes`` -- e.g. ``[2, 8, 1]`` or ``[4, 16, 16, 3]``; the number of
  hidden layers is ``len(layer_sizes) - 2``;
* ``loss`` -- ``"bce"`` (sigmoid output + binary cross-entropy, Exercise 2)
  or ``"cce"`` (softmax output + categorical cross-entropy, Exercises 3-4);
* ``rng`` -- the ``numpy.random.Generator`` used for the initial weights and
  for shuffling the mini-batches, so every run is reproducible.

The forward pass, the backward pass and the update are the *same code* for
both losses. That works because both output/loss pairs share one gradient
at the output pre-activation, dL/du = (y_hat - y) / N (derived in the
report), so the only thing that changes between Exercise 2 and Exercise 3
is which pair is plugged in.

This module only defines functions and classes; the exercise scripts in the
same folder import it.
"""

from __future__ import annotations

import numpy as np


# --8<-- [start:activations-losses]
# -----------------------------------------------------------------------------
# Activations
# -----------------------------------------------------------------------------
def tanh(u: np.ndarray) -> np.ndarray:
    return np.tanh(u)


def tanh_derivative_from_output(h: np.ndarray) -> np.ndarray:
    """d/du tanh(u) = 1 - tanh(u)^2, written in terms of h = tanh(u)."""
    return 1.0 - h ** 2


def sigmoid(u: np.ndarray) -> np.ndarray:
    # Clipping only avoids overflow in exp for very large |u|; it does not
    # change the value in any range the network actually reaches.
    return 1.0 / (1.0 + np.exp(-np.clip(u, -500, 500)))


def softmax(u: np.ndarray) -> np.ndarray:
    """Row-wise softmax. Subtracting the row max leaves the result unchanged
    (it cancels between numerator and denominator) but keeps exp() finite."""
    e = np.exp(u - u.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# -----------------------------------------------------------------------------
# Losses -- each one is an (output activation, loss value) pair
# -----------------------------------------------------------------------------
EPS = 1e-12  # keeps log() away from log(0)


def binary_cross_entropy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """L = -(1/N) sum[y log y_hat + (1 - y) log(1 - y_hat)], y in {0, 1}, shape (N, 1)."""
    y_hat = np.clip(y_hat, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))


def categorical_cross_entropy(y_hat: np.ndarray, y: np.ndarray) -> float:
    """L = -(1/N) sum_n sum_k y_nk log y_hat_nk, y one-hot, shape (N, K)."""
    return float(-np.mean(np.sum(y * np.log(np.clip(y_hat, EPS, 1.0)), axis=1)))


# name -> (output activation, loss function)
LOSSES = {
    "bce": (sigmoid, binary_cross_entropy),
    "cce": (softmax, categorical_cross_entropy),
}
# --8<-- [end:activations-losses]


# --8<-- [start:mlp-class]
class MLP:
    """Fully connected network: tanh on every hidden layer, sigmoid/softmax on the output."""

    def __init__(self, layer_sizes: list[int], loss: str, rng: np.random.Generator):
        if len(layer_sizes) < 3:
            raise ValueError("need at least one hidden layer: [n_in, hidden..., n_out]")
        self.layer_sizes = list(layer_sizes)
        self.output_activation, self.loss_fn = LOSSES[loss]

        # Xavier/Glorot-style initialisation: W ~ N(0, 1/fan_in). It keeps the
        # pre-activations of each layer at roughly unit scale, so tanh starts
        # in its non-saturated region and gradients neither vanish nor explode.
        # W[l] has shape (fan_in, fan_out) so that a batch is X @ W + b.
        self.W = [rng.normal(0.0, np.sqrt(1.0 / n_in), size=(n_in, n_out))
                  for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in layer_sizes[1:]]

    def n_parameters(self) -> int:
        return int(sum(W.size + b.size for W, b in zip(self.W, self.b)))

    # ---- forward pass -------------------------------------------------------
    def forward(self, X: np.ndarray) -> list[np.ndarray]:
        """Return the activations of every layer, input included: [X, h1, ..., y_hat].

        Hidden layer l: u = h_{l-1} W_l + b_l, h_l = tanh(u).
        Output layer:   u = h_{L-1} W_L + b_L, y_hat = sigmoid(u) or softmax(u).
        The list is kept because the backward pass needs every h_l.
        """
        activations = [X]
        h = X
        last = len(self.W) - 1
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            u = h @ W + b
            h = self.output_activation(u) if l == last else tanh(u)
            activations.append(h)
        return activations

    # ---- backward pass ------------------------------------------------------
    def backward(self, activations: list[np.ndarray], y: np.ndarray):
        """Gradients of the mean loss w.r.t. every W_l and b_l, by the chain rule.

        Output layer: for sigmoid+BCE and for softmax+CCE alike,
            delta_L = dL/du_L = (y_hat - y) / N.
        Then, walking from the last layer to the first:
            dL/dW_l = h_{l-1}^T delta_l,    dL/db_l = sum over the batch of delta_l,
            dL/dh_{l-1} = delta_l W_l^T,
            delta_{l-1} = dL/dh_{l-1} * (1 - h_{l-1}^2)      (tanh derivative).
        """
        n = y.shape[0]
        grads_W = [np.empty_like(W) for W in self.W]
        grads_b = [np.empty_like(b) for b in self.b]

        delta = (activations[-1] - y) / n
        for l in range(len(self.W) - 1, -1, -1):
            h_prev = activations[l]
            grads_W[l] = h_prev.T @ delta
            grads_b[l] = delta.sum(axis=0)
            if l > 0:  # no gradient needed w.r.t. the input X itself
                delta = (delta @ self.W[l].T) * tanh_derivative_from_output(h_prev)
        return grads_W, grads_b

    # ---- parameter update ---------------------------------------------------
    def update(self, grads_W, grads_b, eta: float) -> None:
        """Plain gradient descent: theta <- theta - eta * dL/dtheta."""
        for l in range(len(self.W)):
            self.W[l] -= eta * grads_W[l]
            self.b[l] -= eta * grads_b[l]

    # ---- helpers built on the three steps above ----------------------------
    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.loss_fn(self.forward(X)[-1], y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        if p.shape[1] == 1:                      # binary: threshold the sigmoid at 0.5
            return (p[:, 0] >= 0.5).astype(int)
        return p.argmax(axis=1)                  # multi-class: most probable class

    def fit(self, X: np.ndarray, y: np.ndarray, eta: float, epochs: int,
            batch_size: int, rng: np.random.Generator) -> list[float]:
        """Mini-batch gradient descent. Each epoch reshuffles the training set
        with `rng`, then runs forward -> backward -> update once per batch.
        Returns the full-training-set loss measured at the end of every epoch."""
        history = []
        n = X.shape[0]
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                activations = self.forward(X[idx])
                grads_W, grads_b = self.backward(activations, y[idx])
                self.update(grads_W, grads_b, eta)
            history.append(self.loss(X, y))
        return history
# --8<-- [end:mlp-class]


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Integer labels (N,) -> one-hot matrix (N, n_classes)."""
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def accuracy(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    """y holds integer labels, shape (N,)."""
    return float(np.mean(model.predict(X) == y))
