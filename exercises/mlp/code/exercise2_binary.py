"""Exercise 2 -- binary classification that no straight line can do.

Class 0 is one Gaussian cluster at the origin; Class 1 is two clusters on
opposite sides of it, at (3, 3) and (-3, -3). A logistic-regression baseline
(one straight boundary) is compared with the from-scratch MLP in ``mlp.py``.

Randomness: ``make_blobs`` and ``train_test_split`` use ``random_state=42``
as the statement prescribes; a single ``numpy.random.default_rng(42)`` is
created once and used for everything the MLP draws (initial weights and the
mini-batch shuffling of every epoch).

Run from the repository root:

    python docs/exercises/mlp/code/exercise2_binary.py

It prints every number quoted in the report and writes Figures 1-3 to
``docs/exercises/mlp/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp import MLP, accuracy  # noqa: E402  (the shared, from-scratch network)

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# Class colours (validated categorical palette) + a different marker per class,
# so the classes stay distinguishable without relying on colour alone.
COLORS = {0: "#2a78d6", 1: "#eb6834"}
MARKERS = {0: "o", 1: "^"}

# --8<-- [start:ex2-data]
rng = np.random.default_rng(42)

# A -- data, exactly as given in the statement
centers = np.array([[0., 0.], [3., 3.], [-3., -3.]])
X, c = make_blobs(n_samples=[500, 250, 250], centers=centers,
                  cluster_std=1.2, random_state=42)
y = (c > 0).astype(int)      # class 0: 1 cluster | class 1: 2 clusters

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
# --8<-- [end:ex2-data]

print(f"train: {X_train.shape[0]} points {np.bincount(y_train)} | "
      f"test: {X_test.shape[0]} points {np.bincount(y_test)}")

fig, ax = plt.subplots(figsize=(6.4, 5.6))
for cls in (0, 1):
    pts = X[y == cls]
    ax.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.65, color=COLORS[cls],
               marker=MARKERS[cls], label=f"Class {cls} (n={len(pts)})")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Figure 1 - Exercise 2: the 1000 points coloured by class")
ax.legend(loc="upper left", title="Class")
ax.set_aspect("equal")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURES / "figure1.png", dpi=150)
plt.close(fig)

# --8<-- [start:ex2-baseline]
# B -- linear baseline: one straight boundary, fitted on the training set only
baseline = LogisticRegression(random_state=42).fit(X_train, y_train)
baseline_acc = baseline.score(X_test, y_test)
# --8<-- [end:ex2-baseline]
print(f"logistic regression test accuracy: {baseline_acc:.4f}")
print(f"  coef = {np.round(baseline.coef_[0], 4)}, intercept = {baseline.intercept_[0]:.4f}")
p_test = baseline.predict_proba(X_test)[:, 1]
print(f"  predicted P(class 1) on the test set: min {p_test.min():.4f}, max {p_test.max():.4f}")
print(f"  share of test points predicted as class 1: {baseline.predict(X_test).mean():.4f}")

# For contrast with the baseline: a single straight line can at most cut off
# ONE of the two Class 1 clusters -- e.g. the one at (3, 3) with the line
# x1 + x2 = 3, halfway between the centres (0, 0) and (3, 3) -- leaving the
# other Class 1 cluster on the Class 0 side.
one_cluster_pred = (X_test.sum(axis=1) > 3).astype(int)
print(f"line x1 + x2 = 3 test accuracy: {np.mean(one_cluster_pred == y_test):.4f}")

# --8<-- [start:ex2-train]
# C -- the from-scratch MLP: 2 inputs -> 8 tanh units -> 1 sigmoid output, BCE loss
LAYERS = [2, 8, 1]
ETA, EPOCHS, BATCH = 0.1, 300, 32

model = MLP(LAYERS, loss="bce", rng=rng)
loss_history = model.fit(X_train, y_train.reshape(-1, 1).astype(float),
                         eta=ETA, epochs=EPOCHS, batch_size=BATCH, rng=rng)
mlp_acc = accuracy(model, X_test, y_test)
# --8<-- [end:ex2-train]
print(f"MLP {LAYERS}: {model.n_parameters()} parameters, eta={ETA}, epochs={EPOCHS}, batch={BATCH}")
print(f"  train loss: epoch 1 {loss_history[0]:.4f}, epoch 50 {loss_history[49]:.4f}, "
      f"final {loss_history[-1]:.4f}")
print(f"  train accuracy {accuracy(model, X_train, y_train):.4f} | test accuracy {mlp_acc:.4f}")

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(np.arange(1, EPOCHS + 1), loss_history, color=COLORS[0], linewidth=2,
        label=f"MLP {LAYERS} (training set)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary cross-entropy (training set)")
ax.set_title("Figure 2 - Exercise 2: training loss per epoch")
ax.legend(loc="upper right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURES / "figure2.png", dpi=150)
plt.close(fig)

# Figure 3: evaluate the trained network on a dense grid and shade both regions
x1 = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 400)
x2 = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 400)
G1, G2 = np.meshgrid(x1, x2)
grid_pred = model.predict(np.c_[G1.ravel(), G2.ravel()]).reshape(G1.shape)

fig, ax = plt.subplots(figsize=(6.4, 5.8))
ax.contourf(G1, G2, grid_pred, levels=[-0.5, 0.5, 1.5],
            cmap=ListedColormap(["#cde2fb", "#fbd9c9"]))
ax.contour(G1, G2, grid_pred, levels=[0.5], colors="#52514e", linewidths=1.5)
wrong = model.predict(X_test) != y_test
for cls in (0, 1):
    pts = X_test[y_test == cls]
    ax.scatter(pts[:, 0], pts[:, 1], s=22, color=COLORS[cls], marker=MARKERS[cls],
               edgecolors="white", linewidths=0.5, label=f"Test point, class {cls}")
ax.scatter(X_test[wrong, 0], X_test[wrong, 1], s=90, facecolors="none",
           edgecolors="black", linewidths=1.3, label=f"Misclassified (n={wrong.sum()})")
handles, _ = ax.get_legend_handles_labels()
handles += [Patch(color="#cde2fb", label="Region predicted class 0"),
            Patch(color="#fbd9c9", label="Region predicted class 1")]
ax.legend(handles=handles, loc="lower right", fontsize=8, title="Class")
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title(f"Figure 3 - Exercise 2: MLP decision regions, test set (acc. {mlp_acc:.3f})")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIGURES / "figure3.png", dpi=150)
plt.close(fig)

# Shape of the class-1 region along the diagonal x1 = x2 (where both class-1
# clusters sit): where does the prediction switch between 0 and 1?
t = np.linspace(-8, 8, 3201)
diag = model.predict(np.c_[t, t])
switches = t[1:][np.diff(diag) != 0]
print(f"along x1 = x2 the prediction changes at t = {np.round(switches, 2)}")
print(f"  (class 1 for t < {switches[0]:.2f} and t > {switches[-1]:.2f})")
