"""Exercise 3 -- three classes, same network.

The MLP class in ``mlp.py`` is imported and used *unchanged*: the forward
pass, the backward pass and the update step are the same code that trained
the binary network in Exercise 2. Only arguments change -- the layer sizes
(3 output units), ``loss="cce"`` (softmax + categorical cross-entropy) and
the hyperparameters.

Randomness: ``make_classification`` and ``train_test_split`` use
``random_state=42``; one ``numpy.random.default_rng(42)`` drives the MLP's
initial weights and mini-batch shuffling.

Run from the repository root:

    python docs/exercises/mlp/code/exercise3_multiclass.py

It prints every number quoted in the report and writes Figures 4-5.
Exercise 4 imports ``load_data`` and the hyperparameters from this file so
that both exercises are guaranteed to use the same data and budget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp import MLP, accuracy, one_hot  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --8<-- [start:ex3-config]
N_CLASSES = 3
LAYERS_EX3 = [4, 16, 3]          # 4 inputs -> 16 tanh units -> 3 softmax outputs
ETA, EPOCHS, BATCH = 0.05, 300, 32


def load_data():
    """A -- data exactly as given in the statement, 80/20 stratified split."""
    X, y = make_classification(n_samples=1500, n_features=4, n_informative=4,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=2, class_sep=1.0,
                               random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return X, y, X_train, X_test, y_train, y_test


def train(layers, X_train, y_train, rng):
    """Build and train one network; the same call is used by Exercise 4."""
    model = MLP(layers, loss="cce", rng=rng)
    history = model.fit(X_train, one_hot(y_train, N_CLASSES),
                        eta=ETA, epochs=EPOCHS, batch_size=BATCH, rng=rng)
    return model, history
# --8<-- [end:ex3-config]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    # --8<-- [start:ex3-train]
    rng = np.random.default_rng(42)
    X, y, X_train, X_test, y_train, y_test = load_data()

    model, loss_history = train(LAYERS_EX3, X_train, y_train, rng)
    mlp_acc = accuracy(model, X_test, y_test)
    cm = confusion_matrix(y_test, model.predict(X_test), labels=[0, 1, 2])
    # --8<-- [end:ex3-train]

    print(f"train: {X_train.shape[0]} {np.bincount(y_train)} | test: {X_test.shape[0]} "
          f"{np.bincount(y_test)}")
    print(f"MLP {LAYERS_EX3}: {model.n_parameters()} parameters, eta={ETA}, "
          f"epochs={EPOCHS}, batch={BATCH}")
    print(f"  train loss: epoch 1 {loss_history[0]:.4f}, final {loss_history[-1]:.4f}")
    print(f"  train accuracy {accuracy(model, X_train, y_train):.4f} | "
          f"test accuracy {mlp_acc:.4f}")
    print("confusion matrix (rows = true class, columns = predicted class):")
    print(cm)
    off = cm + cm.T
    pairs = {(i, j): int(off[i, j]) for i in range(3) for j in range(i + 1, 3)}
    print(f"  errors per unordered pair (both directions): {pairs}")

    # Baseline: one linear softmax boundary per class (multinomial logistic regression)
    baseline = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    baseline_acc = baseline.score(X_test, y_test)
    print(f"logistic regression test accuracy: {baseline_acc:.4f}")
    print("  baseline confusion matrix:")
    print(confusion_matrix(y_test, baseline.predict(X_test), labels=[0, 1, 2]))

    # --8<-- [start:ex3-overlap]
    # How much do the classes overlap in the 4-D input space? For every point,
    # find its nearest neighbour (Euclidean, plain numpy) among all the other
    # points and record that neighbour's class. A pair of classes whose points
    # often have each other as nearest neighbour is a pair whose clusters touch.
    sq = (X ** 2).sum(axis=1)
    dist2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.fill_diagonal(dist2, np.inf)
    nn_class = y[dist2.argmin(axis=1)]
    nn_table = np.array([[np.sum((y == i) & (nn_class == j)) for j in range(3)]
                         for i in range(3)])
    # --8<-- [end:ex3-overlap]
    print("nearest-neighbour class table, full dataset (row = class, col = class of NN):")
    print(nn_table)
    nn_pairs = {(i, j): int(nn_table[i, j] + nn_table[j, i])
                for i in range(3) for j in range(i + 1, 3)}
    print(f"  cross-class NN links per pair: {nn_pairs}")
    centroids = np.array([X[y == k].mean(axis=0) for k in range(3)])
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  distance between class means {i}-{j}: "
                  f"{np.linalg.norm(centroids[i] - centroids[j]):.4f}")

    # ---- Figure 4: training loss ------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(np.arange(1, EPOCHS + 1), loss_history, color="#2a78d6", linewidth=2,
            label=f"MLP {LAYERS_EX3} (training set)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical cross-entropy (training set)")
    ax.set_title("Figure 4 - Exercise 3: training loss per epoch")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4.png", dpi=150)
    plt.close(fig)

    # ---- Figure 5: confusion matrix on the test set -----------------------
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "#0b0b0b")
    labels = [f"Class {k}" for k in range(3)]
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"Figure 5 - Exercise 3: confusion matrix\n(test set, n={len(y_test)})")
    fig.colorbar(im, ax=ax, label="Number of test points")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5.png", dpi=150)
    plt.close(fig)
