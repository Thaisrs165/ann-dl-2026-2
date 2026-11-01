"""Exercise 2 -- a latent space you can see (d = 2).

Trains the same VAE as Exercise 1 with d = 2 (same data, seed, beta = 1,
optimiser and 20 epochs), reports its test numbers next to the d = 16 ones,
and draws Figures 4-6. Figure 6's latent row needs the d = 16 model of
Exercise 1: it is retrained here with the identical seed and budget, which
reproduces it exactly (the script prints its test numbers to check that).

Run from the repository root:

    python docs/exercises/vae/code/exercise2_latent2d.py

It prints every number quoted in the report and writes Figures 4-6 to
``docs/exercises/vae/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae import cell_centres, evaluate, load_mnist, tile, train_vae  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

# Ten digits: the eight validated categorical hues, then two neutrals. Ten
# classes exceed what colour alone can separate, so each digit is also
# labelled directly at its cluster centre in Figure 4.
DIGIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#6250d6", "#e34948", "#52514e", "#a0682a"]

# Two test images of different digits for Figure 6 (test #2 is the first 1,
# test #18 the first 3 -- see Figure 2).
IDX_A, IDX_B = 2, 18

x_train, x_val, x_test, y_test = load_mnist()

# --8<-- [start:ex2-train]
# Exercise 1's model, rebuilt: same seed and the same calls in the same order
# (train, then evaluate on test), so even the eps drawn for its test ELBO match
model16, hist16 = train_vae(16, x_train, x_val, verbose=False)
r16, k16 = evaluate(model16, x_test)

# A -- same model and budget, only d changes (train_vae re-seeds with 42)
model2, hist2 = train_vae(2, x_train, x_val)
r2, k2 = evaluate(model2, x_test)
print(f"\nTEST d=2 : recon {r2:.2f}  KL {k2:.2f}  negative ELBO {r2 + k2:.2f}")
print(f"TEST d=16: recon {r16:.2f}  KL {k16:.2f}  negative ELBO {r16 + k16:.2f}")
print(f"validation KL d=2: epoch 1 {hist2['val_kl'][0]:.2f}, epoch 20 {hist2['val_kl'][-1]:.2f}")
# --8<-- [end:ex2-train]

model2.eval()
model16.eval()

# ---- per-digit reconstruction, decoding mu (evidence for 2E) ----------------
with torch.no_grad():
    def recon_per_image(model):
        mu, _ = model.encode(x_test)
        x_hat = model.decode(mu).clamp(1e-7, 1 - 1e-7)
        return -(x_test * x_hat.log() + (1 - x_test) * (1 - x_hat).log()).sum(1)
    bce2, bce16 = recon_per_image(model2), recon_per_image(model16)
print("\nper-digit test recon from decode(mu)   digit: d=2 / d=16")
for k in range(10):
    m = y_test == k
    print(f"  {k}: {bce2[m].mean():6.1f} / {bce16[m].mean():6.1f}")

# --8<-- [start:ex2-figures]
with torch.no_grad():
    # B -- every test image encoded to its mean mu in the plane
    mu_test, _ = model2.encode(x_test)

    # C -- 15 x 15 grid at the prior's quantiles: z = Phi^{-1}(q), q in [0.05, 0.95]
    q = np.linspace(0.05, 0.95, 15)
    zs = norm.ppf(q)
    # row r = z2 from top (largest) to bottom, column c = z1 from left to right,
    # so the tile has the same orientation as the scatter of Figure 4
    grid = torch.tensor([[z1, z2] for z2 in zs[::-1] for z1 in zs], dtype=torch.float32)
    x_grid = model2.decode(grid)

    # D -- interpolation between two test images, t = 0, 1/9, ..., 1
    t = torch.linspace(0, 1, 10).unsqueeze(1)
    xa, xb = x_test[IDX_A], x_test[IDX_B]
    pixel_row = (1 - t) * xa + t * xb                       # straight line in pixel space
    mu_a, _ = model16.encode(xa.unsqueeze(0))
    mu_b, _ = model16.encode(xb.unsqueeze(0))
    latent_row = model16.decode((1 - t) * mu_a + t * mu_b)  # straight line in the d=16 latent space
# --8<-- [end:ex2-figures]

with torch.no_grad():
    _, logvar_test = model2.encode(x_test)
print(f"\nd=2 mean posterior sigma per dim: "
      f"{' '.join(f'{v:.3f}' for v in torch.exp(0.5 * logvar_test).mean(0).tolist())}")

mu_np, y_np = mu_test.numpy(), y_test.numpy()
print(f"\nz grid values: {np.round(zs, 3)}")
print(f"mu_test range: z1 [{mu_np[:, 0].min():.2f}, {mu_np[:, 0].max():.2f}], "
      f"z2 [{mu_np[:, 1].min():.2f}, {mu_np[:, 1].max():.2f}]; "
      f"fraction of mu inside the grid square: "
      f"{np.mean((np.abs(mu_np) <= zs[-1]).all(1)):.3f}")

# Overlap in Figure 4, quantified: classify each test mu by its 10 nearest other
# test mus (majority vote) and count which true -> predicted pairs are most common.
d2 = ((mu_np[:, None, :] - mu_np[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(d2, np.inf)
nn10 = np.argsort(d2, axis=1)[:, :10]
pred = np.array([np.bincount(y_np[row], minlength=10).argmax() for row in nn10])
conf = np.zeros((10, 10), int)
np.add.at(conf, (y_np, pred), 1)
print(f"10-NN accuracy in the 2-D mu plane: {np.mean(pred == y_np):.3f}")
pairs = sorted(((conf[i, j] + conf[j, i], i, j) for i in range(10) for j in range(i + 1, 10)),
               reverse=True)
print("most-mixed digit pairs (errors both ways):",
      ", ".join(f"{i}-{j}: {n}" for n, i, j in pairs[:6]))
print("per-digit 10-NN accuracy:", " ".join(f"{k}:{conf[k, k] / conf[k].sum():.2f}" for k in range(10)))

# ---- Figure 4: the encoded test set ------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))
order = np.random.default_rng(42).permutation(len(y_np))   # no digit painted entirely on top
ax.scatter(mu_np[order, 0], mu_np[order, 1], c=[DIGIT_COLORS[k] for k in y_np[order]],
           s=6, alpha=0.6, linewidths=0)
for k in range(10):
    cx, cy = np.median(mu_np[y_np == k], axis=0)
    ax.text(cx, cy, str(k), fontsize=15, fontweight="bold", ha="center", va="center",
            color="#0b0b0b", bbox=dict(boxstyle="circle,pad=0.25", fc="white",
                                       ec=DIGIT_COLORS[k], lw=2.5))
handles = [plt.Line2D([], [], marker="o", ls="", color=DIGIT_COLORS[k], markersize=8,
                      label=f"digit {k}") for k in range(10)]
ax.legend(handles=handles, title="true class", loc="center left", bbox_to_anchor=(1.01, 0.5))
ax.set_title("Figure 4 — test set encoded by the d = 2 VAE (posterior means μ)\n"
             "circled labels mark each digit's median position")
ax.set_xlabel("μ₁ (latent dimension 1)")
ax.set_ylabel("μ₂ (latent dimension 2)")
ax.grid(alpha=0.3)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIGURES / "figure4.png", dpi=150)
plt.close(fig)

# ---- Figure 5: the decoded plane ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10.4))
ax.imshow(tile(x_grid, 15, 15), cmap="gray_r", vmin=0, vmax=1)
ax.set_xticks(cell_centres(15), [f"{z:.2f}" for z in zs], rotation=60)
ax.set_yticks(cell_centres(15), [f"{z:.2f}" for z in zs[::-1]])
ax.set_xlabel("z₁ = Φ⁻¹(q),  q = 0.05 … 0.95")
ax.set_ylabel("z₂ = Φ⁻¹(q),  q = 0.05 … 0.95")
ax.set_title("Figure 5 — decoder of the d = 2 VAE on a 15 × 15 grid of prior quantiles")
fig.tight_layout()
fig.savefig(FIGURES / "figure5.png", dpi=150)
plt.close(fig)

# ---- Figure 6: pixel vs latent interpolation ---------------------------------
fig, ax = plt.subplots(figsize=(11, 3.1))
ax.imshow(tile(torch.cat([pixel_row, latent_row]), 2, 10), cmap="gray_r", vmin=0, vmax=1)
ax.set_xticks(cell_centres(10), [f"{v:.2f}" for v in t.squeeze(1).tolist()])
ax.set_yticks(cell_centres(2), ["pixel space\n(1−t)x_a + t x_b", "latent space, d = 16\ndecode((1−t)μ_a + t μ_b)"])
ax.set_xlabel("interpolation weight t")
ax.set_ylabel("path")
ax.set_title(f"Figure 6 — interpolating test #{IDX_A} (digit {int(y_test[IDX_A])}) "
             f"→ test #{IDX_B} (digit {int(y_test[IDX_B])})")
fig.tight_layout()
fig.savefig(FIGURES / "figure6.png", dpi=150)
plt.close(fig)

print(f"\nFigures written to {FIGURES}")
