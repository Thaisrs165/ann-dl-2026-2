"""Shared VAE code for Exercises 1 and 2: data, model, loss, training loop.

Everything VAE-specific is written by hand here: the reparameterization
(``reparameterize``), both loss terms (``vae_loss``: BCE summed over the 784
pixels and the closed-form KL against N(0, I)), and the sampling done in the
exercise scripts. ``torch.nn`` is used only for the linear layers, ReLU and
sigmoid, and autograd computes the gradients.

Units: every loss number is in nats per image, i.e. summed over pixels (and
over latent dimensions for the KL) and averaged over the images.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torchvision import datasets, transforms

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3


# --8<-- [start:data]
def load_mnist(root: str = "data"):
    """Return (x_train, x_val, x_test, y_test) as flattened float tensors in [0, 1].

    The split is exactly the one in the statement: 50 000 / 10 000 out of the
    60 000 training images, with ``torch.Generator().manual_seed(42)``. The
    10 000 official test images are kept apart and only used for the final
    numbers (and the figures drawn after training).
    """
    train_full = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    test = datasets.MNIST(root, train=False, download=True, transform=transforms.ToTensor())
    train, val = random_split(train_full, [50_000, 10_000],
                              generator=torch.Generator().manual_seed(42))

    # ToTensor() already maps pixels to [0, 1]; dividing the raw uint8 array by
    # 255 gives the same values and avoids 70 000 PIL round-trips. The subsets'
    # .indices are the ones random_split drew above.
    all_x = train_full.data.float().div(255).view(-1, 784)
    x_train = all_x[torch.tensor(train.indices)]
    x_val = all_x[torch.tensor(val.indices)]
    x_test = test.data.float().div(255).view(-1, 784)
    return x_train, x_val, x_test, test.targets.clone()
# --8<-- [end:data]


# --8<-- [start:model]
class VAE(nn.Module):
    """Encoder 784 -> 512 -> 256 (ReLU), heads 256 -> d for mu and log sigma^2,
    decoder d -> 256 -> 512 -> 784 (ReLU on the hidden layers, sigmoid output)."""

    def __init__(self, d: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256, d)       # head for mu
        self.fc_logvar = nn.Linear(256, d)   # head for log sigma^2 (unconstrained, so sigma > 0 always)
        self.decoder = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),   # each output = P(pixel is "on")
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def reparameterize(mu, logvar):
    """z = mu + sigma * eps with eps ~ N(0, I).

    The randomness lives in eps, which does not depend on the parameters, so
    gradients flow through mu and sigma = exp(logvar / 2) as usual.
    """
    sigma = torch.exp(0.5 * logvar)
    eps = torch.randn_like(sigma)
    return mu + sigma * eps


def vae_loss(x_hat, x, mu, logvar, beta: float = 1.0):
    """Return (total, recon, kl), each averaged over the images of the batch.

    recon = -sum_j [x_j log x_hat_j + (1 - x_j) log(1 - x_hat_j)]   (784 pixels)
    kl    = 1/2 sum_i (mu_i^2 + sigma_i^2 - 1 - log sigma_i^2)      (d latent dims)

    reduction="sum" then / batch size keeps the BCE summed over pixels;
    reduction="mean" would also divide it by 784 and silently weaken it
    relative to the KL.
    """
    n = x.shape[0]
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / n
    kl = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar) / n
    return recon + beta * kl, recon, kl
# --8<-- [end:model]


# --8<-- [start:train]
@torch.no_grad()
def evaluate(model, x, batch_size: int = 1000):
    """Average recon and KL (nats/image) over a whole set, with z sampled as in training."""
    model.eval()
    recon_sum = kl_sum = 0.0
    for i in range(0, len(x), batch_size):
        xb = x[i:i + batch_size]
        x_hat, mu, logvar = model(xb)
        _, recon, kl = vae_loss(x_hat, xb, mu, logvar)
        recon_sum += recon.item() * len(xb)   # undo the per-batch average
        kl_sum += kl.item() * len(xb)
    return recon_sum / len(x), kl_sum / len(x)


def train_vae(d: int, x_train, x_val, epochs: int = EPOCHS, verbose: bool = True):
    """Build a VAE with latent size d (after torch.manual_seed(42)) and train it.

    Returns the model and a history dict with, per epoch, the training recon/KL
    (average over the epoch's mini-batches) and the validation recon/KL
    (evaluated at the end of the epoch). Only the validation set is looked at
    during training; the test set never enters this function.
    """
    torch.manual_seed(42)   # fixes the initial weights, the shuffles and every eps
    model = VAE(d)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    hist = {k: [] for k in ("train_recon", "train_kl", "val_recon", "val_kl")}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train))   # new shuffle every epoch
        recon_sum = kl_sum = 0.0
        for i in range(0, len(x_train), BATCH_SIZE):
            xb = x_train[perm[i:i + BATCH_SIZE]]
            x_hat, mu, logvar = model(xb)
            loss, recon, kl = vae_loss(x_hat, xb, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            recon_sum += recon.item() * len(xb)
            kl_sum += kl.item() * len(xb)

        hist["train_recon"].append(recon_sum / len(x_train))
        hist["train_kl"].append(kl_sum / len(x_train))
        v_recon, v_kl = evaluate(model, x_val)
        hist["val_recon"].append(v_recon)
        hist["val_kl"].append(v_kl)
        if verbose:
            print(f"[d={d:2d}] epoch {epoch:2d}  train recon {hist['train_recon'][-1]:7.2f}  "
                  f"KL {hist['train_kl'][-1]:6.2f}  |  val recon {v_recon:7.2f}  KL {v_kl:6.2f}"
                  f"  ({time.time() - t0:5.1f}s)")
    return model, hist
# --8<-- [end:train]


def tile(images, nrows: int, ncols: int, pad: int = 2, gutter: float = 0.15):
    """Arrange a (nrows*ncols, 784) tensor of images into one 2-D array, with a
    ``pad``-pixel light-grey gutter between the 28x28 cells (for plotting only)."""
    imgs = images.detach().view(nrows, ncols, 28, 28).cpu()
    cell = 28 + pad
    canvas = torch.full((nrows * cell + pad, ncols * cell + pad), gutter)
    for r in range(nrows):
        for c in range(ncols):
            canvas[pad + r * cell: pad + r * cell + 28, pad + c * cell: pad + c * cell + 28] = imgs[r, c]
    return canvas.numpy()


def cell_centres(n: int, pad: int = 2):
    """Pixel coordinate of the centre of each of the n cells laid out by ``tile``."""
    return [pad + i * (28 + pad) + 13.5 for i in range(n)]
