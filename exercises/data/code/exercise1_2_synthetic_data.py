"""Exercises 1 and 2 — synthetic data for neural-network preprocessing.

Exercise 1 studies a 2D, 4-class problem and how spreading the classes apart
(or squeezing them together) changes how separable they are with straight
decision boundaries. Exercise 2 studies two 5D, 2-class problems: shifted
correlated Gaussians (Dataset I) and concentric spherical shells (Dataset II).

A single ``numpy.random.default_rng(42)`` instance is created once and reused
for every draw in both exercises, in the order the sections appear below, so
that a fresh run reproduces every number reported in the report exactly.

Run from the repository root:

    python docs/exercises/data/code/exercise1_2_synthetic_data.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --8<-- [start:preamble]
rng = np.random.default_rng(42)
# --8<-- [end:preamble]

# =============================================================================
# Exercise 1 — 2D point clouds
# =============================================================================

# --8<-- [start:ex1-a]
MEANS = np.array(
    [
        [2, 3],
        [5, 6],
        [8, 1],
        [15, 4],
    ],
    dtype=float,
)

STDS = np.array(
    [
        [0.8, 2.5],
        [1.2, 1.9],
        [0.9, 0.9],
        [0.5, 2.0],
    ],
    dtype=float,
)

N_PER_CLASS_EX1 = 100
CLASS_LABELS_EX1 = [0, 1, 2, 3]


def generate_clouds(scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Draw 100 points per class with std = STDS * scale, means unchanged."""
    xs, ys = [], []
    for label in CLASS_LABELS_EX1:
        std = STDS[label] * scale
        # Each feature is drawn independently with its own mean/std.
        x1 = rng.normal(MEANS[label, 0], std[0], size=N_PER_CLASS_EX1)
        x2 = rng.normal(MEANS[label, 1], std[1], size=N_PER_CLASS_EX1)
        xs.append(np.column_stack([x1, x2]))
        ys.append(np.full(N_PER_CLASS_EX1, label))
    return np.vstack(xs), np.concatenate(ys)


def nearest_mean_labels(points: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Assign each point to the label of its nearest fixed mean (Euclidean)."""
    dists = np.linalg.norm(points[:, None, :] - means[None, :, :], axis=2)
    return np.argmin(dists, axis=1)


def figure1_clouds_and_boundaries() -> None:
    """Figure 1: 400 points at s = 1, class means, and nearest-mean decision boundaries."""
    X, y = generate_clouds(scale=1.0)

    pad = 2.0
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    zz = nearest_mean_labels(grid, MEANS).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    # Nearest-mean decision regions: boundaries sit exactly at half-integer
    # contour levels because zz only takes the integer values 0..3.
    ax.contour(
        xx, yy, zz,
        levels=[0.5, 1.5, 2.5],
        colors="black",
        linestyles="--",
        linewidths=1.3,
        zorder=2,
    )
    ax.plot([], [], color="black", linestyle="--", linewidth=1.3,
             label="Estimated linear boundaries (nearest-mean regions)")

    for label in CLASS_LABELS_EX1:
        pts = X[y == label]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=18, alpha=0.75, color=colors[label],
            label=f"Class {label}", zorder=3,
        )

    ax.scatter(
        MEANS[:, 0], MEANS[:, 1],
        marker="*", s=380, color="gold", edgecolor="black", linewidth=1.2,
        label="Class means", zorder=4,
    )

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("Figure 1 — Four Gaussian classes (s = 1.0) with estimated linear boundaries")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
# --8<-- [end:ex1-a]


# --8<-- [start:ex1-b]
def mixing_rate(points: np.ndarray, labels: np.ndarray, means: np.ndarray) -> float:
    """Fraction of points whose nearest-mean assignment disagrees with the true label."""
    predicted = nearest_mean_labels(points, means)
    return float(np.mean(predicted != labels))


def pairwise_separation_ratios() -> dict[tuple[int, int], float]:
    """r_ij = ||mu_i - mu_j|| / (sigma_bar_i + sigma_bar_j) for every class pair, s = 1."""
    sigma_bar = STDS.mean(axis=1)
    ratios = {}
    for i in CLASS_LABELS_EX1:
        for j in CLASS_LABELS_EX1:
            if i < j:
                dist = np.linalg.norm(MEANS[i] - MEANS[j])
                ratios[(i, j)] = float(dist / (sigma_bar[i] + sigma_bar[j]))
    return ratios


def figure2_spread_study(scales: list[float]) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Figure 2: one subplot per scale, shared axis limits across all four."""
    datasets = {scale: generate_clouds(scale) for scale in scales}

    all_x = np.concatenate([X[:, 0] for X, _ in datasets.values()])
    all_y = np.concatenate([X[:, 1] for X, _ in datasets.values()])
    pad = 1.5
    xlim = (all_x.min() - pad, all_x.max() + pad)
    ylim = (all_y.min() - pad, all_y.max() + pad)

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharex=True, sharey=True)
    for ax, scale in zip(axes, scales):
        X, y = datasets[scale]
        for label in CLASS_LABELS_EX1:
            pts = X[y == label]
            ax.scatter(pts[:, 0], pts[:, 1], s=12, alpha=0.75, color=colors[label],
                       label=f"Class {label}")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel("$x_1$")
        ax.set_title(f"s = {scale}")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("$x_2$")
    axes[-1].legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle("Figure 2 — Class spread for scale $s \\in \\{0.5, 1.0, 2.0, 4.0\\}$ (shared axes)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure2.png", dpi=150)
    plt.close(fig)
    return datasets


def figure3_mixing_rate(datasets: dict[float, tuple[np.ndarray, np.ndarray]]) -> dict[float, float]:
    """Figure 3: mixing rate vs. scale, using the datasets already generated for Figure 2."""
    scales = sorted(datasets)
    rates = {scale: mixing_rate(*datasets[scale], MEANS) for scale in scales}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(scales, [rates[s] for s in scales], marker="o", color="#d95f02",
            label="Mixing rate (nearest-mean disagreement)")
    ax.set_xlabel("Scale $s$")
    ax.set_ylabel("Mixing rate")
    ax.set_title("Figure 3 — Mixing rate vs. spread scale")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure3.png", dpi=150)
    plt.close(fig)
    return rates
# --8<-- [end:ex1-b]


# =============================================================================
# Exercise 2 — 5D datasets
# =============================================================================

N_PER_CLASS_EX2 = 500

# --8<-- [start:ex2-a]
MU_A = np.array([0, 0, 0, 0, 0], dtype=float)
SIGMA_A = np.array(
    [
        [1.0, 0.8, 0.1, 0.0, 0.0],
        [0.8, 1.0, 0.3, 0.0, 0.0],
        [0.1, 0.3, 1.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 1.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 1.0],
    ]
)

MU_B = np.array([1.5, 1.5, 1.5, 1.5, 1.5], dtype=float)
SIGMA_B = np.array(
    [
        [1.5, -0.7, 0.2, 0.0, 0.0],
        [-0.7, 1.5, 0.4, 0.0, 0.0],
        [0.2, 0.4, 1.5, 0.6, 0.0],
        [0.0, 0.0, 0.6, 1.5, 0.3],
        [0.0, 0.0, 0.0, 0.3, 1.5],
    ]
)


def generate_shifted_gaussians() -> tuple[np.ndarray, np.ndarray]:
    """Dataset I: Class A ~ N(mu_a, sigma_a), Class B ~ N(mu_b, sigma_b), 500 each."""
    X_a = rng.multivariate_normal(MU_A, SIGMA_A, size=N_PER_CLASS_EX2)
    X_b = rng.multivariate_normal(MU_B, SIGMA_B, size=N_PER_CLASS_EX2)
    X = np.vstack([X_a, X_b])
    y = np.concatenate([np.zeros(N_PER_CLASS_EX2), np.ones(N_PER_CLASS_EX2)])
    return X, y
# --8<-- [end:ex2-a]


# --8<-- [start:ex2-b]
RADIUS_C = (2.0, 0.4)  # (mean, std) of Normal radius
RADIUS_D = (5.0, 0.4)


def generate_concentric_shells() -> tuple[np.ndarray, np.ndarray]:
    """Dataset II: Class C radius ~ N(2.0, 0.4), Class D radius ~ N(5.0, 0.4), 500 each."""
    def shell(radius_mean: float, radius_std: float) -> np.ndarray:
        directions = rng.standard_normal(size=(N_PER_CLASS_EX2, 5))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        radii = rng.normal(radius_mean, radius_std, size=N_PER_CLASS_EX2)
        return directions * radii[:, None]

    X_c = shell(*RADIUS_C)
    X_d = shell(*RADIUS_D)
    X = np.vstack([X_c, X_d])
    y = np.concatenate([np.zeros(N_PER_CLASS_EX2), np.ones(N_PER_CLASS_EX2)])
    return X, y
# --8<-- [end:ex2-b]


# --8<-- [start:ex2-c]
def figure4_pca(dataset1: tuple[np.ndarray, np.ndarray], dataset2: tuple[np.ndarray, np.ndarray]) -> dict:
    """Figure 4: separate 2-component PCA projection for each dataset, side by side."""
    (X1, y1), (X2, y2) = dataset1, dataset2

    pca1 = PCA(n_components=2)
    Z1 = pca1.fit_transform(X1)
    pca2 = PCA(n_components=2)
    Z2 = pca2.fit_transform(X2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, Z, y, title, labels in zip(
        axes, [Z1, Z2], [y1, y2],
        ["Dataset I — shifted Gaussians (Classes A/B)", "Dataset II — concentric shells (Classes C/D)"],
        [("A", "B"), ("C", "D")],
    ):
        for cls, name, color in zip([0, 1], labels, ["#1b9e77", "#d95f02"]):
            pts = Z[y == cls]
            ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.7, color=color, label=f"Class {name}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("Figure 4 — PCA projections of the two 5D datasets")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4.png", dpi=150)
    plt.close(fig)

    return {
        "dataset1": {"pc1": pca1.explained_variance_ratio_[0], "pc2": pca1.explained_variance_ratio_[1]},
        "dataset2": {"pc1": pca2.explained_variance_ratio_[0], "pc2": pca2.explained_variance_ratio_[1]},
    }


def figure5_radius_histograms(dataset1: tuple[np.ndarray, np.ndarray], dataset2: tuple[np.ndarray, np.ndarray]) -> dict:
    """Figure 5: class centers, center distance, and overlaid radius-from-origin histograms."""
    (X1, y1), (X2, y2) = dataset1, dataset2

    center_a, center_b = X1[y1 == 0].mean(axis=0), X1[y1 == 1].mean(axis=0)
    center_c, center_d = X2[y2 == 0].mean(axis=0), X2[y2 == 1].mean(axis=0)
    dist1 = float(np.linalg.norm(center_a - center_b))
    dist2 = float(np.linalg.norm(center_c - center_d))

    radius1 = np.linalg.norm(X1, axis=1)
    radius2 = np.linalg.norm(X2, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bins1 = np.linspace(radius1.min(), radius1.max(), 30)
    axes[0].hist(radius1[y1 == 0], bins=bins1, alpha=0.6, color="#1b9e77", label="Class A")
    axes[0].hist(radius1[y1 == 1], bins=bins1, alpha=0.6, color="#d95f02", label="Class B")
    axes[0].set_title("Dataset I — radius from origin")
    axes[0].set_xlabel(r"$\|x\|$")
    axes[0].set_ylabel("Count")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.25)

    bins2 = np.linspace(radius2.min(), radius2.max(), 30)
    axes[1].hist(radius2[y2 == 0], bins=bins2, alpha=0.6, color="#7570b3", label="Class C")
    axes[1].hist(radius2[y2 == 1], bins=bins2, alpha=0.6, color="#e7298a", label="Class D")
    axes[1].set_title("Dataset II — radius from origin")
    axes[1].set_xlabel(r"$\|x\|$")
    axes[1].set_ylabel("Count")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Figure 5 — Radius-from-origin distributions per class")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5.png", dpi=150)
    plt.close(fig)

    return {
        "center_a": center_a, "center_b": center_b, "dist1": dist1,
        "center_c": center_c, "center_d": center_d, "dist2": dist2,
        "radius1": radius1, "radius2": radius2,
        "radius_c": radius2[y2 == 0], "radius_d": radius2[y2 == 1],
    }
# --8<-- [end:ex2-c]


# --8<-- [start:ex2-d]
def suggest_radial_threshold(radius_c: np.ndarray, radius_d: np.ndarray) -> tuple[float, float]:
    """Midpoint between the observed mean radii of the core (C) and shell (D) classes.

    Returns the threshold both as a radius (compare against ||x||) and as the
    squared radius (compare against f(x) = sum_i x_i^2 = ||x||^2), which is the
    quantity actually used by the proposed radial separation rule.
    """
    threshold_radius = float((radius_c.mean() + radius_d.mean()) / 2)
    return threshold_radius, threshold_radius ** 2
# --8<-- [end:ex2-d]


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # --- Exercise 1 ---
    figure1_clouds_and_boundaries()

    ratios = pairwise_separation_ratios()
    print("Pairwise separation ratios (s = 1.0):")
    for (i, j), r in sorted(ratios.items(), key=lambda kv: kv[1]):
        print(f"  r_{i}{j} = {r:.4f}")
    smallest_pair, smallest_ratio = min(ratios.items(), key=lambda kv: kv[1])
    print(f"Smallest ratio: pair {smallest_pair} = {smallest_ratio:.4f}")
    print(f"Smallest ratio at s = 2.0 (= s=1 value / 2): {smallest_ratio / 2:.4f}")

    scales = [0.5, 1.0, 2.0, 4.0]
    datasets_ex1 = figure2_spread_study(scales)
    rates = figure3_mixing_rate(datasets_ex1)
    print("Mixing rates:")
    for s in scales:
        print(f"  s = {s}: mixing rate = {rates[s]:.4f}")

    # --- Exercise 2 ---
    dataset1 = generate_shifted_gaussians()
    dataset2 = generate_concentric_shells()

    pca_stats = figure4_pca(dataset1, dataset2)
    print("PCA explained variance:")
    for name, stats in pca_stats.items():
        total = stats["pc1"] + stats["pc2"]
        print(f"  {name}: PC1={stats['pc1']:.4f}, PC2={stats['pc2']:.4f}, sum={total:.4f}")

    radius_stats = figure5_radius_histograms(dataset1, dataset2)
    print(f"Distance between centers — Dataset I: {radius_stats['dist1']:.4f}")
    print(f"Distance between centers — Dataset II: {radius_stats['dist2']:.4f}")
    print(f"Mean radius — Class C: {radius_stats['radius_c'].mean():.4f}")
    print(f"Mean radius — Class D: {radius_stats['radius_d'].mean():.4f}")

    threshold_radius, threshold_sq = suggest_radial_threshold(
        radius_stats["radius_c"], radius_stats["radius_d"]
    )
    print(f"Suggested radial threshold (mean radius): {threshold_radius:.4f}")
    print(f"Suggested radial threshold (f(x) = ||x||^2): {threshold_sq:.4f}")


if __name__ == "__main__":
    main()
