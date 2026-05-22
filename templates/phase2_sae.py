"""
fca-interp phase2_sae — Galois embedding of SAE features (headline theorem)
+ P2 (structural polysemanticity tracks ground truth).

Track A+B. On a toy model with KNOWN planted features:
  1. Train a small SAE on the hidden activations.
  2. Build the FCA closed-concept set (Track A) on the same activations.
  3. For each SAE feature (binarised firing pattern), test whether its support
     corresponds to a closed concept of the FCA lattice (Galois embedding).
     Report the fraction of SAE features that embed, and relate non-embedding
     features to SAE quality (reconstruction contribution / dead-ness).
  4. P2: label neurons polysemantic by ground truth (a hidden unit is
     polysemantic if it has large weight to >1 planted feature) and compare to
     structural polysemanticity (Definition 1, join-reducibility of its
     attribute concepts). Report F1.

Outputs to /kaggle/working/:
  - phase2_sae_results.json
  - phase2_sae_embedding.png
"""

import os, sys
def _find_core(name):
    for dp, _, fs in os.walk("/kaggle/input"):
        if f"{name}.py" in fs: return dp
    return None
sys.path.insert(0, _find_core("fca_core"))
import fca_core as core
import numpy as np
import torch, torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

torch.manual_seed(0); np.random.seed(0)
t0 = time.time()

# ---------------------------------------------------------------------------
# Toy model with known planted features (Elhage et al.)
# ---------------------------------------------------------------------------
class ToyAE(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(k, d) * 0.5)
        self.b = nn.Parameter(torch.zeros(k))
    def forward(self, x):
        h = x @ self.W
        return torch.relu(h @ self.W.T + self.b), h

def train_toy(k, d, sp, steps=3000, lr=1e-2, decay=0.8, seed=0, n=10000):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ToyAE(k, d); imp = torch.tensor([decay**i for i in range(k)])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        mask = (torch.rand(n, k) < sp).float(); x = mask * torch.rand(n, k)
        y, _ = m(x); loss = ((x - y)**2 * imp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

K, D, SP = 8, 5, 0.15
model = train_toy(K, D, SP, seed=1)

# Probe activations + ground-truth active-feature record per sample
rng = np.random.default_rng(42)
N = 2000
mask = (rng.random((N, K)) < SP)
x = (mask * rng.random((N, K))).astype(np.float32)
with torch.no_grad():
    h = torch.relu(torch.from_numpy(x) @ model.W).numpy()      # (N, D) hidden
core.format_result("toy_K_features", K)
core.format_result("toy_D_hidden", D)

# ---------------------------------------------------------------------------
# Train a small SAE on the hidden activations h  (overcomplete dictionary)
# ---------------------------------------------------------------------------
class SAE(nn.Module):
    def __init__(self, d_in, d_dict, l1=1e-3):
        super().__init__()
        self.enc = nn.Linear(d_in, d_dict)
        self.dec = nn.Linear(d_dict, d_in, bias=False)
        self.l1 = l1
    def forward(self, x):
        f = torch.relu(self.enc(x))
        return self.dec(f), f

def train_sae(acts, d_dict, steps=4000, lr=1e-3, l1=1e-3, seed=0):
    torch.manual_seed(seed)
    sae = SAE(acts.shape[1], d_dict, l1)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    X = torch.from_numpy(acts)
    for _ in range(steps):
        recon, f = sae(X)
        loss = ((recon - X)**2).sum(1).mean() + l1 * f.abs().sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, f = sae(X)
    return sae, f.numpy()

D_DICT = 2 * K   # overcomplete
sae, feats = train_sae(h, D_DICT, seed=0)
# binarise SAE feature firings
feat_fire = feats > 1e-6
feat_density = feat_fire.mean(0)
n_dead = int((feat_density < 1e-4).sum())
core.format_result("trackB_sae_dict_size", D_DICT)
core.format_result("trackB_sae_dead_features", n_dead)
core.format_result("trackB_sae_mean_density", f"{feat_density.mean():.3f}")

# ---------------------------------------------------------------------------
# Galois embedding test: does each SAE feature's support match an FCA concept?
# Build the FCA closed-concept set on h; represent each concept by its EXTENT
# (set of probe samples). An SAE feature embeds if its firing set equals (or
# closely matches) some concept extent.
# ---------------------------------------------------------------------------
concepts = core.frequent_closed_concepts(h, scaling="s0", s0_threshold="zero", min_support=0.01)
concept_extents = [set(c["extent"].tolist()) for c in concepts]
core.format_result("trackA_n_closed_concepts", len(concepts))

def best_jaccard_to_concepts(fire_set, extents):
    if not fire_set:
        return 0.0
    best = 0.0
    for e in extents:
        inter = len(fire_set & e)
        if inter == 0:
            continue
        j = inter / len(fire_set | e)
        if j > best:
            best = j
    return best

embed_scores = []
for c in range(D_DICT):
    if feat_density[c] < 1e-4:
        continue  # skip dead
    fire_set = set(np.where(feat_fire[:, c])[0].tolist())
    embed_scores.append({
        "feature": c,
        "density": float(feat_density[c]),
        "best_jaccard": best_jaccard_to_concepts(fire_set, concept_extents),
    })

EMBED_THRESH = 0.5
n_alive = len(embed_scores)
n_embed = sum(1 for e in embed_scores if e["best_jaccard"] >= EMBED_THRESH)
frac_embed = n_embed / max(n_alive, 1)
core.format_result("theorem_n_alive_sae_features", n_alive)
core.format_result("theorem_n_embedded", n_embed)
core.format_result("theorem_frac_embedded", f"{frac_embed:.3f}")

# Relate non-embedding features to density (are they spurious/rare?)
embedded = [e["density"] for e in embed_scores if e["best_jaccard"] >= EMBED_THRESH]
non_embedded = [e["density"] for e in embed_scores if e["best_jaccard"] < EMBED_THRESH]
core.format_result(
    "theorem_density_embedded_vs_not",
    f"embedded_mean={np.mean(embedded) if embedded else 0:.3f} "
    f"non_embedded_mean={np.mean(non_embedded) if non_embedded else 0:.3f}",
)

# ---------------------------------------------------------------------------
# P2: structural polysemanticity vs ground truth
# Ground truth: hidden unit j is "polysemantic" if it has substantial weight
# (|W[feature, j]|) to more than one planted feature. W is (K, D); column j is
# the readout of hidden unit j across the K features.
# ---------------------------------------------------------------------------
Wd = model.W.detach().numpy()    # (K, D)
# normalise columns; count features with weight above a fraction of the max
col = np.abs(Wd)                  # (K, D)
gt_poly = np.zeros(D, dtype=bool)
for j in range(D):
    w = col[:, j]
    thresh = 0.5 * w.max()
    gt_poly[j] = (w >= thresh).sum() > 1

struct = core.neuron_structural_polysemanticity(h, scaling="s1", n_quantiles=3)
struct_poly = np.array(struct["poly_mask"], dtype=bool)

tp = int((gt_poly & struct_poly).sum())
fp = int((~gt_poly & struct_poly).sum())
fn = int((gt_poly & ~struct_poly).sum())
prec = tp / max(tp + fp, 1)
rec = tp / max(tp + fn, 1)
f1 = 2 * prec * rec / max(prec + rec, 1e-9)
core.format_result("P2_ground_truth_poly_count", int(gt_poly.sum()))
core.format_result("P2_structural_poly_count", int(struct_poly.sum()))
core.format_result("P2_precision", f"{prec:.3f}")
core.format_result("P2_recall", f"{rec:.3f}")
core.format_result("P2_f1", f"{f1:.3f}")
core.format_result("P2_met", f1 >= 0.8)

# ---------------------------------------------------------------------------
# Plot: SAE feature embedding-quality histogram
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist([e["best_jaccard"] for e in embed_scores], bins=20, range=(0, 1))
ax.axvline(EMBED_THRESH, ls="--", color="red", label=f"embed threshold={EMBED_THRESH}")
ax.set_xlabel("best Jaccard(SAE feature support, FCA concept extent)")
ax.set_ylabel("# SAE features")
ax.set_title(f"Galois embedding: {n_embed}/{n_alive} SAE features map to FCA concepts")
ax.legend()
plt.tight_layout()
plt.savefig("/kaggle/working/phase2_sae_embedding.png", dpi=120, bbox_inches="tight")
plt.close()

results = {
    "meta": {"K": K, "D": D, "sparsity": SP, "sae_dict": D_DICT},
    "theorem": {
        "n_closed_concepts": len(concepts),
        "n_alive_sae_features": n_alive,
        "n_embedded": n_embed,
        "frac_embedded": frac_embed,
        "embed_threshold": EMBED_THRESH,
    },
    "p2": {"precision": prec, "recall": rec, "f1": f1,
            "gt_poly": int(gt_poly.sum()), "struct_poly": int(struct_poly.sum())},
    "elapsed_sec": time.time() - t0,
    "experiment_complete": True,
}
core.format_result("elapsed_sec", f"{results['elapsed_sec']:.1f}")
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase2_sae_results.json")
print("Done.")
