"""Exercise 4 -- does a second hidden layer help?

A controlled comparison: the data, the split, the learning rate, the number
of epochs and the batch size are all imported from Exercise 3, so the ONLY
difference between the two networks is depth:

    Exercise 3: [4, 16, 3]        -- 1 hidden layer
    Exercise 4: [4, 16, 16, 3]    -- 2 hidden layers (same width, one more layer)

Each comparison is run with several seeds. For a given seed, both networks
get a fresh ``numpy.random.default_rng(seed)``, so the pair shares the same
source of randomness for initial weights and mini-batch order. Seed 42
reproduces Exercise 3's network exactly.

Run from the repository root:

    python docs/exercises/mlp/code/exercise4_depth.py

It prints the per-seed and mean numbers quoted in the report and writes Figure 6.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exercise3_multiclass import (BATCH, EPOCHS, ETA, LAYERS_EX3,  # noqa: E402
                                  load_data, train)
from mlp import accuracy  # noqa: E402

FIGURES = Path(__file__).resolve().parents[1] / "figures"

# --8<-- [start:ex4-compare]
LAYERS_EX4 = [4, 16, 16, 3]
SEEDS = [42, 43, 44]

X, y, X_train, X_test, y_train, y_test = load_data()   # same data as Exercise 3

results = {"Ex3": [], "Ex4": []}
curves = {"Ex3": {}, "Ex4": {}}
for seed in SEEDS:
    for name, layers in (("Ex3", LAYERS_EX3), ("Ex4", LAYERS_EX4)):
        rng = np.random.default_rng(seed)          # same seed for both members of the pair
        model, history = train(layers, X_train, y_train, rng)
        results[name].append({
            "seed": seed,
            "params": model.n_parameters(),
            "final_loss": history[-1],
            "train_acc": accuracy(model, X_train, y_train),
            "test_acc": accuracy(model, X_test, y_test),
        })
        curves[name][seed] = history
# --8<-- [end:ex4-compare]

print(f"budget: eta={ETA}, epochs={EPOCHS}, batch={BATCH}; seeds={SEEDS}")
for name, layers in (("Ex3", LAYERS_EX3), ("Ex4", LAYERS_EX4)):
    print(f"\n{name} {layers} ({len(layers) - 2} hidden layer(s), "
          f"{results[name][0]['params']} parameters)")
    for r in results[name]:
        print(f"  seed {r['seed']}: final train loss {r['final_loss']:.4f} | "
              f"train acc {r['train_acc']:.4f} | test acc {r['test_acc']:.4f}")
    for key in ("final_loss", "train_acc", "test_acc"):
        vals = np.array([r[key] for r in results[name]])
        print(f"  mean {key}: {vals.mean():.4f} (std {vals.std(ddof=1):.4f})")

diffs = np.array([b["test_acc"] - a["test_acc"]
                  for a, b in zip(results["Ex3"], results["Ex4"])])
print(f"\ntest-accuracy difference Ex4 - Ex3 per seed: {np.round(diffs, 4)}")
print(f"mean difference: {diffs.mean():+.4f} ({diffs.mean() * 100:+.2f} percentage points)")

# Where does each network's training loss stand at a few checkpoints (seed-mean)?
for epoch in (10, 50, 100, 300):
    m3 = np.mean([curves["Ex3"][s][epoch - 1] for s in SEEDS])
    m4 = np.mean([curves["Ex4"][s][epoch - 1] for s in SEEDS])
    print(f"  epoch {epoch:3d}: mean train loss Ex3 {m3:.4f} | Ex4 {m4:.4f}")

# ---- Figure 6: both loss curves on the same axes -------------------------
colors = {"Ex3": "#2a78d6", "Ex4": "#eb6834"}
labels = {"Ex3": f"Exercise 3 {LAYERS_EX3} (1 hidden layer)",
          "Ex4": f"Exercise 4 {LAYERS_EX4} (2 hidden layers)"}
epochs = np.arange(1, EPOCHS + 1)
fig, ax = plt.subplots(figsize=(7.0, 4.6))
for name in ("Ex3", "Ex4"):
    for seed in SEEDS[1:]:                         # other seeds: thin, faint
        ax.plot(epochs, curves[name][seed], color=colors[name], linewidth=0.8, alpha=0.35)
    ax.plot(epochs, curves[name][SEEDS[0]], color=colors[name], linewidth=2,
            label=f"{labels[name]}, seed {SEEDS[0]}")
ax.plot([], [], color="#52514e", linewidth=0.8, alpha=0.5,
        label=f"thin lines: seeds {', '.join(map(str, SEEDS[1:]))}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Categorical cross-entropy (training set)")
ax.set_title("Figure 6 - Exercise 4: training loss, 1 vs 2 hidden layers")
ax.legend(loc="upper right", fontsize=8.5, title="Model")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURES / "figure6.png", dpi=150)
plt.close(fig)
