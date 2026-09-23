"""Exercises 1 and 2 -- a from-scratch perceptron on separable and overlapping data.

Exercise 1 trains a single-layer perceptron on two well-separated Gaussian
clouds -- the case the perceptron was designed for. Exercise 2 reuses the
*exact same* implementation, unchanged, on two heavily overlapping clouds --
the case it cannot solve -- and tracks a "pocket" (best-so-far) set of
weights alongside whatever the loop's last iterate happens to be.

A single ``numpy.random.default_rng(42)`` instance is created once and reused
for every draw in both exercises, in the order the sections appear below:
Exercise 1's two classes, then Exercise 1's initial weights (reused verbatim
for the eta = 1.0 rerun in D2), then Exercise 2's two classes, then Exercise
2's initial weights. The zero-init algebraic check in D3 draws nothing from
`rng` (both runs start deterministically at w = 0, b = 0).

Run from the repository root:

    python docs/exercises/perceptron/code/exercise1_2_perceptron.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --8<-- [start:preamble]
rng = np.random.default_rng(42)
# --8<-- [end:preamble]

# =============================================================================
# Shared perceptron implementation (Exercise 1B) -- reused unchanged in Ex. 2B
# =============================================================================

# --8<-- [start:perceptron-impl]
def step(z: np.ndarray) -> np.ndarray:
    """Heaviside step: 1 where z >= 0, 0 otherwise. Labels/predictions live in {0, 1}."""
    return np.where(z >= 0, 1, 0)


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Vectorized prediction y_hat = step(w . x + b) for a batch of samples."""
    return step(X @ w + b)


def accuracy(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    return float(np.mean(predict(X, w, b) == y))


def train_perceptron(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray,
    b_init: float,
    eta: float,
    max_epochs: int = 100,
    track_pocket: bool = False,
) -> dict:
    """Online single-layer perceptron, error-driven update, 0/1 labels.

    Per-sample update: e = y - y_hat in {-1, 0, +1};
    w <- w + eta*e*x, b <- b + eta*e (no-op when e == 0).
    Stops at the first update-free epoch, or after `max_epochs`.

    When `track_pocket` is True, the *only* addition to the loop is: after
    every update, check whether the new (w, b) beats the best full-dataset
    accuracy seen so far and, if so, copy it into the pocket. This is the
    pocket algorithm used, unchanged, for Exercise 2.
    """
    w = np.array(w_init, dtype=float).copy()
    b = float(b_init)
    n = X.shape[0]

    epoch_accuracy: list[float] = []   # accuracy of the *current* weights, after every epoch
    pocket_accuracy: list[float] = []  # best-so-far accuracy, after every epoch
    pocket = {"w": w.copy(), "b": b, "accuracy": accuracy(X, y, w, b), "epoch": 0}

    epoch = 0
    n_updates = 0
    for epoch in range(1, max_epochs + 1):
        n_updates = 0
        for i in range(n):
            xi, yi = X[i], y[i]
            y_hat = 1 if (xi @ w + b) >= 0 else 0
            e = yi - y_hat
            if e != 0:
                w = w + eta * e * xi
                b = b + eta * e
                n_updates += 1
                if track_pocket:
                    acc = accuracy(X, y, w, b)
                    if acc > pocket["accuracy"]:
                        pocket = {"w": w.copy(), "b": b, "accuracy": acc, "epoch": epoch}
        epoch_accuracy.append(accuracy(X, y, w, b))
        pocket_accuracy.append(pocket["accuracy"])
        if n_updates == 0:
            break

    return {
        "w": w, "b": b, "epochs": epoch, "n_updates_last_epoch": n_updates,
        "epoch_accuracy": epoch_accuracy, "pocket_accuracy": pocket_accuracy,
        "final_accuracy": epoch_accuracy[-1],
        "pocket": pocket if track_pocket else None,
    }
# --8<-- [end:perceptron-impl]


# =============================================================================
# Plotting helpers (shared by both exercises)
# =============================================================================

COLORS = {0: "#1b9e77", 1: "#d95f02"}


def scatter_classes(ax, X: np.ndarray, y: np.ndarray) -> None:
    for cls in (0, 1):
        pts = X[y == cls]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.6, color=COLORS[cls],
                   label=f"Class {cls}", zorder=2)


def plot_boundary(ax, w: np.ndarray, b: float, x_range: tuple[float, float],
                   color: str, label: str, linestyle: str = "-") -> None:
    """Draw w . x + b = 0 over x_range, handling a near-vertical boundary."""
    if abs(w[1]) > 1e-9:
        x1 = np.linspace(*x_range, 200)
        x2 = -(w[0] * x1 + b) / w[1]
        ax.plot(x1, x2, color=color, linestyle=linestyle, linewidth=2.2, label=label, zorder=3)
    else:
        x0 = -b / w[0]
        ax.axvline(x0, color=color, linestyle=linestyle, linewidth=2.2, label=label, zorder=3)


def mark_misclassified(ax, X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float,
                       label_prefix: str = "Misclassified") -> int:
    y_hat = predict(X, w, b)
    wrong = X[y_hat != y]
    ax.scatter(wrong[:, 0], wrong[:, 1], s=70, facecolors="none", edgecolors="black",
               linewidths=1.5, marker="o", label=f"{label_prefix} (n={len(wrong)})", zorder=4)
    return int(len(wrong))


# =============================================================================
# Exercise 1 -- separable data
# =============================================================================

# --8<-- [start:ex1-a]
EX1_MEAN0, EX1_COV0 = [1.5, 1.5], [[0.5, 0.0], [0.0, 0.5]]
EX1_MEAN1, EX1_COV1 = [5.0, 5.0], [[0.5, 0.0], [0.0, 0.5]]
N_PER_CLASS = 1000


def generate_two_classes(rng, mean0, cov0, mean1, cov1, n=N_PER_CLASS):
    """Class 0 ~ N(mean0, cov0), Class 1 ~ N(mean1, cov1), n samples each."""
    X0 = rng.multivariate_normal(mean0, cov0, size=n)
    X1 = rng.multivariate_normal(mean1, cov1, size=n)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y


def figure1_ex1_scatter(X: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figure 1 -- Exercise 1: separable data (2000 points)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1.png", dpi=150)
    plt.close(fig)
# --8<-- [end:ex1-a]


# --8<-- [start:ex1-c]
def figure2_ex1_boundary(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> int:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    x_range = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plot_boundary(ax, w, b, x_range, color="black", label="Decision boundary $w \\cdot x + b = 0$")
    n_wrong = mark_misclassified(ax, X, y, w, b)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figure 2 -- Exercise 1: decision boundary ($\\eta = 0.01$)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2.png", dpi=150)
    plt.close(fig)
    return n_wrong


def figure3_ex1_accuracy(epoch_accuracy: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    epochs = np.arange(1, len(epoch_accuracy) + 1)
    ax.plot(epochs, epoch_accuracy, marker="o", markersize=3, color="#1b9e77",
            label="Full-dataset accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 3 -- Exercise 1: accuracy vs. epoch ($\\eta = 0.01$)")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3.png", dpi=150)
    plt.close(fig)
# --8<-- [end:ex1-c]


# --8<-- [start:ex1-d]
def zero_init_scaling_check(X: np.ndarray, y: np.ndarray, eta1: float, eta2: float,
                            max_epochs: int = 100) -> dict:
    """Empirically confirms the D3 algebraic argument: from w=0, b=0, two runs with
    different eta produce weights that differ only by the constant factor eta2/eta1,
    with identical epoch counts and an identical decision boundary."""
    r1 = train_perceptron(X, y, w_init=np.zeros(2), b_init=0.0, eta=eta1, max_epochs=max_epochs)
    r2 = train_perceptron(X, y, w_init=np.zeros(2), b_init=0.0, eta=eta2, max_epochs=max_epochs)
    ratio_w = r2["w"] / r1["w"]
    ratio_b = r2["b"] / r1["b"] if r1["b"] != 0 else np.nan
    return {"run1": r1, "run2": r2, "ratio_w": ratio_w, "ratio_b": ratio_b,
            "expected_ratio": eta2 / eta1}
# --8<-- [end:ex1-d]


# =============================================================================
# Exercise 2 -- overlapping data
# =============================================================================

# --8<-- [start:ex2-a]
EX2_MEAN0, EX2_COV0 = [3.0, 3.0], [[1.5, 0.0], [0.0, 1.5]]
EX2_MEAN1, EX2_COV1 = [4.0, 4.0], [[1.5, 0.0], [0.0, 1.5]]


def figure4_ex2_scatter(X: np.ndarray, y: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter_classes(ax, X, y)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figure 4 -- Exercise 2: overlapping data (2000 points)")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4.png", dpi=150)
    plt.close(fig)
# --8<-- [end:ex2-a]


# --8<-- [start:ex2-b]
def train_ex2(X: np.ndarray, y: np.ndarray, w0: np.ndarray) -> dict:
    """Exercise 2 training call: the exact Exercise 1B loop, only track_pocket=True is new."""
    return train_perceptron(X, y, w_init=w0, b_init=0.0, eta=0.01,
                            max_epochs=100, track_pocket=True)
# --8<-- [end:ex2-b]


# --8<-- [start:ex2-c]
def figure5_ex2_boundaries(X: np.ndarray, y: np.ndarray, w_final: np.ndarray, b_final: float,
                           w_pocket: np.ndarray, b_pocket: float) -> int:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    scatter_classes(ax, X, y)
    x_range = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plot_boundary(ax, w_final, b_final, x_range, color="black",
                  label="Final boundary", linestyle="--")
    plot_boundary(ax, w_pocket, b_pocket, x_range, color="#7570b3",
                  label="Pocket (best-so-far) boundary", linestyle="-")
    # Misclassified points are marked relative to the pocket boundary: the final
    # boundary misclassifies roughly half of all 2000 points (see D1), so marking
    # against it would blanket the plot and carry no information.
    n_wrong = mark_misclassified(ax, X, y, w_pocket, b_pocket, label_prefix="Misclassified (pocket)")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figure 5 -- Exercise 2: final vs. pocket decision boundaries")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5.png", dpi=150)
    plt.close(fig)
    return n_wrong


def figure6_ex2_accuracy(epoch_accuracy: list[float], pocket_accuracy: list[float]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    epochs = np.arange(1, len(epoch_accuracy) + 1)
    ax.plot(epochs, epoch_accuracy, color="black", alpha=0.7, linewidth=1.2,
            label="Current-weights accuracy")
    ax.plot(epochs, pocket_accuracy, color="#7570b3", linewidth=2.2,
            label="Pocket (best-so-far) accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Figure 6 -- Exercise 2: current vs. pocket accuracy vs. epoch")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure6.png", dpi=150)
    plt.close(fig)
# --8<-- [end:ex2-c]


# --8<-- [start:ex2-d]
def best_linear_separator_accuracy(X: np.ndarray, y: np.ndarray, n_theta: int = 360) -> float:
    """Brute-force accuracy of the best possible straight-line classifier on (X, y).

    For each candidate direction theta in [0, pi), points are projected onto
    that direction; the optimal split point along the projection (for either
    label orientation) is found in O(n log n) via a sorted cumulative count.
    Independent numpy-only check used purely for analysis in Exercise 2D -- it
    is not the perceptron and is never used as, or in place of, the trained model.
    """
    n = len(y)
    best_acc = 0.0
    for theta in np.linspace(0.0, np.pi, n_theta, endpoint=False):
        direction = np.array([np.cos(theta), np.sin(theta)])
        proj = X @ direction
        order = np.argsort(proj)
        y_sorted = y[order]
        cum_pos = np.concatenate([[0], np.cumsum(y_sorted == 1)])
        cum_neg = np.concatenate([[0], np.cumsum(y_sorted == 0)])
        total_pos, total_neg = cum_pos[-1], cum_neg[-1]
        # Split at index k: first k points -> class 0, rest -> class 1 (orientation A),
        # or the reverse (orientation B). Try both, at every possible split point.
        correct_a = cum_neg + (total_pos - cum_pos)
        correct_b = cum_pos + (total_neg - cum_neg)
        best_acc = max(best_acc, correct_a.max() / n, correct_b.max() / n)
    return float(best_acc)
# --8<-- [end:ex2-d]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # ----------------------------- Exercise 1 -----------------------------
    X1, y1 = generate_two_classes(rng, EX1_MEAN0, EX1_COV0, EX1_MEAN1, EX1_COV1)
    figure1_ex1_scatter(X1, y1)

    w0 = rng.normal(0, 0.01, size=2)  # reused, unchanged, for the eta=1.0 rerun in D2
    b0 = 0.0
    print(f"Exercise 1 initial weights: w0={w0}, |w0|={np.linalg.norm(w0):.4f}, b0={b0}")

    result_001 = train_perceptron(X1, y1, w0, b0, eta=0.01, max_epochs=100)
    print(f"\n[Ex1, eta=0.01] final w={result_001['w']}, final b={result_001['b']:.4f}")
    print(f"[Ex1, eta=0.01] epochs={result_001['epochs']}, final accuracy={result_001['final_accuracy']:.4f}")

    n_wrong_ex1 = figure2_ex1_boundary(X1, y1, result_001["w"], result_001["b"])
    print(f"[Ex1, eta=0.01] misclassified points: {n_wrong_ex1} / {len(y1)}")
    figure3_ex1_accuracy(result_001["epoch_accuracy"])

    # D1: updates per epoch, to show the decreasing trend directly
    # (re-run bookkeeping done inline in train_perceptron isn't exposed per-epoch,
    #  so this quick pass recomputes just the update counts for reporting)
    def updates_per_epoch(X, y, w_init, b_init, eta, max_epochs=100):
        w = np.array(w_init, dtype=float).copy()
        b = float(b_init)
        counts = []
        for _ in range(max_epochs):
            n_upd = 0
            for i in range(len(X)):
                xi, yi = X[i], y[i]
                y_hat = 1 if (xi @ w + b) >= 0 else 0
                e = yi - y_hat
                if e != 0:
                    w = w + eta * e * xi
                    b = b + eta * e
                    n_upd += 1
            counts.append(n_upd)
            if n_upd == 0:
                break
        return counts

    upd_counts_001 = updates_per_epoch(X1, y1, w0, b0, eta=0.01)
    print(f"[Ex1, eta=0.01] updates per epoch: {upd_counts_001}")

    # D2: eta = 1.0 rerun, same initial weights
    result_1 = train_perceptron(X1, y1, w0, b0, eta=1.0, max_epochs=100)
    print(f"\n[Ex1, eta=1.0] final w={result_1['w']}, final b={result_1['b']:.4f}")
    print(f"[Ex1, eta=1.0] epochs={result_1['epochs']}, final accuracy={result_1['final_accuracy']:.4f}")

    dir_001 = result_001["w"] / np.linalg.norm(result_001["w"])
    dir_1 = result_1["w"] / np.linalg.norm(result_1["w"])
    cos_sim = float(np.dot(dir_001, dir_1))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_sim, -1.0, 1.0))))
    print(f"[Ex1] direction eta=0.01: {dir_001}, |w|={np.linalg.norm(result_001['w']):.4f}")
    print(f"[Ex1] direction eta=1.0:  {dir_1}, |w|={np.linalg.norm(result_1['w']):.4f}")
    print(f"[Ex1] cosine similarity between directions: {cos_sim:.6f} ({angle_deg:.3f} degrees)")

    # D3: zero-init algebraic scaling check
    zic = zero_init_scaling_check(X1, y1, eta1=0.01, eta2=1.0, max_epochs=100)
    print(f"\n[Ex1, zero-init] eta1=0.01: epochs={zic['run1']['epochs']}, w={zic['run1']['w']}, b={zic['run1']['b']:.6f}")
    print(f"[Ex1, zero-init] eta2=1.00: epochs={zic['run2']['epochs']}, w={zic['run2']['w']}, b={zic['run2']['b']:.6f}")
    print(f"[Ex1, zero-init] w ratio (run2/run1): {zic['ratio_w']}, b ratio: {zic['ratio_b']:.4f}, expected: {zic['expected_ratio']:.4f}")

    # ----------------------------- Exercise 2 -----------------------------
    X2, y2 = generate_two_classes(rng, EX2_MEAN0, EX2_COV0, EX2_MEAN1, EX2_COV1)
    figure4_ex2_scatter(X2, y2)

    w0_2 = rng.normal(0, 0.01, size=2)
    print(f"\nExercise 2 initial weights: w0={w0_2}, b0=0.0")

    result_pocket = train_ex2(X2, y2, w0_2)
    pocket = result_pocket["pocket"]
    print(f"\n[Ex2] final w={result_pocket['w']}, final b={result_pocket['b']:.4f}, final accuracy={result_pocket['final_accuracy']:.4f}")
    print(f"[Ex2] pocket w={pocket['w']}, pocket b={pocket['b']:.4f}, pocket accuracy={pocket['accuracy']:.4f}, at epoch={pocket['epoch']}")
    print(f"[Ex2] epochs run: {result_pocket['epochs']} (cap reached: {result_pocket['epochs'] == 100})")

    n_wrong_ex2 = figure5_ex2_boundaries(X2, y2, result_pocket["w"], result_pocket["b"],
                                          pocket["w"], pocket["b"])
    print(f"[Ex2] pocket boundary misclassified points: {n_wrong_ex2} / {len(y2)}")
    figure6_ex2_accuracy(result_pocket["epoch_accuracy"], result_pocket["pocket_accuracy"])

    # ||x|| for the D1 hint
    mean_norm_x2 = float(np.mean(np.linalg.norm(X2, axis=1)))
    print(f"[Ex2] mean ||x|| over the dataset: {mean_norm_x2:.4f}")

    best_acc = best_linear_separator_accuracy(X2, y2, n_theta=360)
    print(f"[Ex2] brute-force best straight-line accuracy (numpy-only check): {best_acc:.4f}")


if __name__ == "__main__":
    main()
