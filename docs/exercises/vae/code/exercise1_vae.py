"""Exercise 1 -- a VAE on MNIST with latent dimension d = 16.

Trains the model in ``vae.py`` (Adam, lr 1e-3, batch 128, 20 epochs), logging
the reconstruction and KL terms separately on train and validation, then
reports the test numbers and draws Figures 1-3.

Randomness: the 50k/10k split uses ``torch.Generator().manual_seed(42)``;
``torch.manual_seed(42)`` is called right before the model is built, which
fixes the initial weights, the shuffles and every eps of training. Figure 3
draws its prior samples after ``torch.manual_seed(0)``.

Run from the repository root (MNIST is downloaded to ./data the first time):

    python docs/exercises/vae/code/exercise1_vae.py

It prints every number quoted in the report and writes Figures 1-3 to
``docs/exercises/vae/figures/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae import EPOCHS, cell_centres, evaluate, load_mnist, tile, train_vae  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)

TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"   # categorical slots 1 and 2
D = 16

# --8<-- [start:ex1-train]
# A -- data (test set only touched after training)
x_train, x_val, x_test, y_test = load_mnist()
print(f"train {tuple(x_train.shape)}, val {tuple(x_val.shape)}, test {tuple(x_test.shape)}")

# C -- train, then the final test numbers
model, hist = train_vae(D, x_train, x_val)
test_recon, test_kl = evaluate(model, x_test)
print(f"\nTEST (d={D}): recon {test_recon:.2f}  KL {test_kl:.2f}  "
      f"negative ELBO {test_recon + test_kl:.2f} nats/image")
print(f"validation KL: epoch 1 {hist['val_kl'][0]:.2f}, epoch {EPOCHS} {hist['val_kl'][-1]:.2f}")
# --8<-- [end:ex1-train]

for k in ("train_recon", "train_kl", "val_recon", "val_kl"):
    print(f"{k:12s}", " ".join(f"{v:6.2f}" for v in hist[k]))

# ---- Figure 1: the two terms per epoch ----------------------------------------
epochs = np.arange(1, EPOCHS + 1)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, term, label in zip(axes, ("recon", "kl"), ("Reconstruction (BCE)", "KL divergence")):
    ax.plot(epochs, hist[f"train_{term}"], color=TRAIN_COLOR, linewidth=2, marker="o",
            markersize=4, label="training (mean over epoch)")
    ax.plot(epochs, hist[f"val_{term}"], color=VAL_COLOR, linewidth=2, marker="s",
            markersize=4, label="validation (end of epoch)")
    ax.set_title(f"{label} per epoch (d = {D})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("nats per image")
    ax.set_xticks(epochs[::2] if EPOCHS > 10 else epochs)
    ax.grid(alpha=0.3)
    ax.legend()
fig.suptitle("Figure 1 — the two loss terms, logged separately")
fig.tight_layout()
fig.savefig(FIGURES / "figure1.png", dpi=150)
plt.close(fig)

# --8<-- [start:ex1-figures]
model.eval()
with torch.no_grad():
    # Figure 2 -- first test image of each digit, reconstructed from the MEAN mu
    first_idx = [int((y_test == k).nonzero()[0]) for k in range(10)]
    x_first = x_test[first_idx]
    mu, _ = model.encode(x_first)
    x_rec = model.decode(mu)                 # decode(mu): no sampling of z

    # Figure 3 -- 64 samples from the prior z ~ N(0, I), decoded
    torch.manual_seed(0)
    z = torch.randn(64, D)
    x_samp = model.decode(z)
# --8<-- [end:ex1-figures]
print("first test index of each digit 0-9:", first_idx)

# ---- posterior statistics on the test set (evidence for 1E) ------------------
with torch.no_grad():
    mu_t, logvar_t = model.encode(x_test)
    sigma_t = torch.exp(0.5 * logvar_t)
    kl_dim = 0.5 * (mu_t.pow(2) + logvar_t.exp() - 1 - logvar_t).mean(0)
    # how far do the 64 prior samples of Figure 3 land from the nearest encoded
    # test image, compared with how far test codes are from each other?
    d_prior = torch.cdist(z, mu_t).min(1).values
    d_data = torch.cdist(mu_t[:64], mu_t[64:]).min(1).values
print("KL per latent dim:", " ".join(f"{v:.2f}" for v in kl_dim.tolist()))
print("mean sigma per dim:", " ".join(f"{v:.2f}" for v in sigma_t.mean(0).tolist()))
print("std of mu per dim:", " ".join(f"{v:.2f}" for v in mu_t.std(0).tolist()))
print(f"active dims (KL > 0.1): {(kl_dim > 0.1).sum().item()} of {D}; "
      f"mean sigma over active dims {sigma_t[:, kl_dim > 0.1].mean():.3f}")
print(f"median distance to nearest test mu: prior samples {d_prior.median():.2f}, "
      f"test codes {d_data.median():.2f}")

fig, ax = plt.subplots(figsize=(11, 2.9))
ax.imshow(tile(torch.cat([x_first, x_rec]), 2, 10), cmap="gray_r", vmin=0, vmax=1)
ax.set_xticks(cell_centres(10), [f"{k}\n(#{i})" for k, i in enumerate(first_idx)])
ax.set_yticks(cell_centres(2), ["original", "decode(μ)"])
ax.set_xlabel("digit (test index)")
ax.set_ylabel("row")
ax.set_title(f"Figure 2 — first test image of each digit and its reconstruction (d = {D})")
fig.tight_layout()
fig.savefig(FIGURES / "figure2.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 7.4))
ax.imshow(tile(x_samp, 8, 8), cmap="gray_r", vmin=0, vmax=1)
ax.set_xticks(cell_centres(8), range(1, 9))
ax.set_yticks(cell_centres(8), range(1, 9))
ax.set_xlabel("sample column")
ax.set_ylabel("sample row")
ax.set_title(f"Figure 3 — 64 images decoded from z ~ N(0, I)\n(d = {D}, torch.manual_seed(0))")
fig.tight_layout()
fig.savefig(FIGURES / "figure3.png", dpi=150)
plt.close(fig)

print(f"\nFigures written to {FIGURES}")
