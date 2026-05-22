"""
fca-interp phase1_mnist — FCA on a real trained MLP (Track A).

Trains a small ReLU MLP on MNIST, takes penultimate-layer activations on a
held-out probe set, builds the S0/S1 concept lattice, and asks: do the
join-irreducible concepts align with digit classes better than k-means
clusters of the same activations?

Metric: for each join-irreducible extent (or k-means cluster), assign the
majority digit label; report mean purity (fraction of the extent that is the
majority class) weighted by extent size, and the number of distinct digit
classes "covered" as majority by some irreducible. FCA structure is canonical
(no k to choose) — k-means is given k = n_join_irreducibles for a fair fight.

Outputs to /kaggle/working/:
  - phase1_mnist_results.json
  - phase1_mnist_purity.png
"""

import os
import sys

def _find_core(name):
    for dirpath, _, filenames in os.walk("/kaggle/input"):
        if f"{name}.py" in filenames:
            return dirpath
    return None

_core_path = _find_core("fca_core")
if _core_path is None:
    raise ImportError("fca_core.py not found under /kaggle/input/")
sys.path.insert(0, _core_path)

import fca_core as core
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

print("=" * 70)
print("PHASE 1 — MNIST MLP: FCA join-irreducibles vs k-means")
print("=" * 70)

t0 = time.time()
torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(root="/kaggle/working/data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST(root="/kaggle/working/data", train=False, download=True, transform=tfm)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)

# ---------------------------------------------------------------------------
# Model: 784 -> 128 -> 32 (penultimate) -> 10
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, d_pen=32):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, d_pen)
        self.fc3 = nn.Linear(d_pen, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        pen = F.relu(self.fc2(x))   # penultimate activations
        return self.fc3(pen), pen

D_PEN = 32
model = MLP(D_PEN)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 3
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        logits, _ = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  epoch {epoch+1}/{EPOCHS} done")

# test accuracy
model.eval()
correct = 0
with torch.no_grad():
    for xb, yb in torch.utils.data.DataLoader(test_ds, batch_size=512):
        logits, _ = model(xb)
        correct += (logits.argmax(1) == yb).sum().item()
acc = correct / len(test_ds)
core.format_result("trackA_mnist_test_acc", f"{acc:.4f}")

# ---------------------------------------------------------------------------
# Probe activations (subset of test set for tractable lattice)
# ---------------------------------------------------------------------------
N_PROBE = 1500
probe_loader = torch.utils.data.DataLoader(test_ds, batch_size=N_PROBE, shuffle=True)
xb, yb = next(iter(probe_loader))
with torch.no_grad():
    _, pen = model(xb)
acts = pen.numpy()
labels_true = yb.numpy()
print(f"  probe activations: {acts.shape}, {N_PROBE} samples")

# ---------------------------------------------------------------------------
# FCA: iceberg concept set (frequent closed concepts) -> extents -> purity
# The P3 diagnostic established that individual join-irreducibles are not a
# stable feature unit; the canonical, reproducible unit is the iceberg concept
# set (frequent closed concepts, proposal §5). We compare its extents' digit
# purity against k-means with k matched to the number of concepts found.
# ---------------------------------------------------------------------------
def purity_metrics(extents, labels_true):
    """Size-weighted mean purity and number of distinct majority classes."""
    extents = [e for e in extents if len(e) > 0]
    if not extents:
        return {"weighted_purity": 0.0, "n_majority_classes": 0, "n_groups": 0,
                "mean_best_f1": 0.0}
    total = 0
    weighted = 0.0
    maj_classes = set()
    for ext in extents:
        ys = labels_true[ext]
        vals, counts = np.unique(ys, return_counts=True)
        maj_classes.add(int(vals[counts.argmax()]))
        weighted += counts.max()
        total += len(ys)
    # FCA-appropriate metric: for each digit class, the best F1 achievable by
    # ANY single group's extent (does the method discover a class-aligned
    # group?). Mean over the 10 classes. This does not penalise the method for
    # also having coarse groups, unlike weighted purity.
    classes = np.unique(labels_true)
    best_f1s = []
    for cl in classes:
        cls_mask = (labels_true == cl)
        n_cls = cls_mask.sum()
        best = 0.0
        for ext in extents:
            tp = cls_mask[ext].sum()
            if tp == 0:
                continue
            prec = tp / len(ext)
            rec = tp / n_cls
            f1 = 2 * prec * rec / (prec + rec)
            if f1 > best:
                best = f1
        best_f1s.append(best)
    return {
        "weighted_purity": float(weighted / max(total, 1)),
        "n_majority_classes": len(maj_classes),
        "n_groups": len(extents),
        "mean_best_f1": float(np.mean(best_f1s)),
    }


results = {"meta": {"d_pen": D_PEN, "n_probe": N_PROBE, "test_acc": acc}, "fca": {}, "kmeans": {}}

# Iceberg support threshold: concepts must cover ≥ 0.5% of the probe set.
# (S1 generates fine-grained concepts; use a lower floor so the set is non-empty.)
ICEBERG_SUPPORT = 0.005

for tag, sc, kw in [("s0_zero", "s0", {"s0_threshold": "zero"}),
                     ("s1_q3",   "s1", {"n_quantiles": 3})]:
    concepts = core.frequent_closed_concepts(acts, scaling=sc, min_support=ICEBERG_SUPPORT, **kw)
    exts = [c["extent"] for c in concepts]
    pm = purity_metrics(exts, labels_true)
    results["fca"][tag] = {**pm, "support_threshold": ICEBERG_SUPPORT}
    core.format_result(
        f"trackA_fca_{tag}",
        f"n_concepts={pm['n_groups']} mean_best_f1={pm['mean_best_f1']:.3f} "
        f"weighted_purity={pm['weighted_purity']:.3f} classes={pm['n_majority_classes']}/10",
    )

    # k-means baseline with k = number of iceberg concepts (fair, matched k)
    k = max(2, min(pm["n_groups"], N_PROBE // 5))
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(acts)
    km_exts = [np.where(km.labels_ == c)[0] for c in range(k)]
    km_pm = purity_metrics(km_exts, labels_true)
    results["kmeans"][tag] = {"k": k, **km_pm}
    core.format_result(
        f"trackA_kmeans_k{k}_for_{tag}",
        f"mean_best_f1={km_pm['mean_best_f1']:.3f} "
        f"weighted_purity={km_pm['weighted_purity']:.3f}",
    )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
tags = list(results["fca"].keys())
fca_pur = [results["fca"][t]["mean_best_f1"] for t in tags]
km_pur = [results["kmeans"][t]["mean_best_f1"] for t in tags]
x = np.arange(len(tags)); w = 0.35
ax.bar(x - w/2, fca_pur, w, label="FCA closed concepts")
ax.bar(x + w/2, km_pur, w, label="k-means (matched k)")
ax.set_xticks(x); ax.set_xticklabels(tags)
ax.set_ylabel("mean best-F1 per digit class")
ax.set_title("MNIST penultimate layer: FCA vs k-means")
ax.legend(); ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("/kaggle/working/phase1_mnist_purity.png", dpi=120, bbox_inches="tight")
plt.close()

results["elapsed_sec"] = time.time() - t0
results["experiment_complete"] = True
core.format_result("elapsed_sec", f"{results['elapsed_sec']:.1f}")
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase1_mnist_results.json")
print("\nDone.")
