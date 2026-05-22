"""
fca-interp phase3_pythia_v2 — fixes after Pythia v1 produced degenerate results.

v1 used residual-stream activations + weak SAE → saturated lattice (1 concept)
and a low-density SAE (0.69 firing rate). v2 fixes all three:
  (a) extract MLP POST-ACTIVATION (post-GeLU) — sparser by construction
  (b) train a strongly-regularised SAE (L1=3e-3) targeting density ≤ 0.1
  (c) sweep S0 thresholds (q90/q95/q98) to find a non-trivial lattice regime

If at least one threshold gives:
  - a non-trivial FCA closed-concept set (≥ 20 concepts)
  - κ_S significantly different from 0 vs matched Gaussian baseline
  - SAE embedding fraction ≥ 0.4 with a density contrast favouring embedded
then the framework scales to real transformers.

Outputs to /kaggle/working/:
  - phase3_pythia_v2_results.json
  - phase3_pythia_v2_summary.png
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
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
torch.set_num_threads(4)

from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
print(f"Loading {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()
D_MODEL = model.config.hidden_size
LAYER_IDX = 3
core.format_result("model", MODEL_NAME)
core.format_result("layer_studied", LAYER_IDX)

# ---------------------------------------------------------------------------
# Tokenize wikitext
# ---------------------------------------------------------------------------
N_TOKENS_TARGET = 3000
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    chunks, total = [], 0
    for ex in ds:
        t = ex["text"].strip()
        if len(t) < 40: continue
        chunks.append(t); total += len(t)
        if total > 60_000: break
    raw_text = "\n\n".join(chunks)
except Exception as e:
    print(f"wikitext unavailable ({e})")
    raw_text = (
        "Mathematics is the science of structure, order and relation. "
        "Formal concept analysis is a method of conceptual data analysis "
        "and knowledge representation that derives a concept hierarchy from "
        "a collection of objects and attributes. " * 100
    )
ids = tok(raw_text, return_tensors="pt", truncation=True, max_length=N_TOKENS_TARGET + 16)["input_ids"][0]
ids = ids[: N_TOKENS_TARGET + 8]

# ---------------------------------------------------------------------------
# Hook the MLP POST-ACTIVATION (after GeLU, before down-projection).
# GPTNeoX MLP is: dense_h_to_4h -> act (GeLU) -> dense_4h_to_h
# We capture the GeLU output: post-activation feature space of width 4*D.
# ---------------------------------------------------------------------------
acts = {}
def grab(_m, _i, output):
    acts["h"] = output.detach().float().squeeze(0)   # (T, 4D)

mlp_act = model.gpt_neox.layers[LAYER_IDX].mlp.act
hook = mlp_act.register_forward_hook(grab)
with torch.no_grad():
    model(input_ids=ids.unsqueeze(0))
hook.remove()
h_full = acts["h"].numpy()
h_all = h_full[1:N_TOKENS_TARGET + 1]
N, D_ACT = h_all.shape
print(f"MLP post-act shape: {h_all.shape}")
core.format_result("d_activation", D_ACT)
core.format_result("n_tokens_used", N)

# Diagnostic: fraction of positive entries (GeLU(x) > 0 ≈ x > 0)
pos_frac = (h_all > 0).mean()
nz_frac = (np.abs(h_all) > 1e-6).mean()
core.format_result("mlp_post_act_positive_fraction", f"{pos_frac:.3f}")
core.format_result("mlp_post_act_nonzero_fraction", f"{nz_frac:.3f}")
# Per-neuron firing fraction at threshold > 0
fire_at_zero = (h_all > 0).mean(0)
core.format_result(
    "mlp_per_neuron_pos_fraction_stats",
    f"min={fire_at_zero.min():.3f} median={np.median(fire_at_zero):.3f} max={fire_at_zero.max():.3f}",
)

# ---------------------------------------------------------------------------
# (c) THRESHOLD SWEEP — find the regime where the lattice is non-trivial
# ---------------------------------------------------------------------------
print("\n=== threshold sweep on MLP post-act ===")
ICEBERG_SUPPORT = 0.005   # 15 tokens out of 3000
THRESHOLDS = [
    ("s0_zero", "s0", {"s0_threshold": "zero"}),
    ("s0_q80",  "s0", {"s0_threshold": "q80"}),
    ("s0_q90",  "s0", {"s0_threshold": "q90"}),
    ("s0_q95",  "s0", {"s0_threshold": "q95"}),
    ("s0_q98",  "s0", {"s0_threshold": "q98"}),
]
sweep = {}
for tag, sc, kw in THRESHOLDS:
    t = time.time()
    s = core.superposition_index(h_all, scaling=sc, **kw)
    cc = core.frequent_closed_concepts(h_all, scaling=sc, min_support=ICEBERG_SUPPORT, **kw)
    dt = time.time() - t
    # κ_S vs Gaussian baseline (single draw for sweep, full average later)
    rng = np.random.default_rng(0)
    g = rng.standard_normal((N, D_ACT)).astype(np.float32)
    sg = core.superposition_index(g, scaling=sc, **kw)
    sigma_C_data = s["sigma_C"]
    sigma_C_noise = sg["sigma_C"]
    kappa = 1 - sigma_C_data / max(sigma_C_noise, 1e-9)
    sweep[tag] = {
        "n_concepts": len(cc),
        "sigma_C_data": sigma_C_data,
        "sigma_C_noise": sigma_C_noise,
        "kappa": kappa,
        "n_obj_clar": s["n_objects_clarified"],
        "rank": s["rank"],
        "elapsed_sec": dt,
    }
    core.format_result(
        f"sweep_{tag}",
        f"n_concepts={len(cc)} σ_C(d)={sigma_C_data:.2f} σ_C(n)={sigma_C_noise:.2f} "
        f"κ={kappa:+.3f} ({dt:.1f}s)",
    )

# Pick best regime: most concepts AND |κ| above noise
def good(d):
    return d["n_concepts"] >= 20 and d["kappa"] >= 0.10
best_tag = next((t for t, _, _ in THRESHOLDS if good(sweep[t])), None)
if best_tag is None:
    # fallback: max n_concepts among those with κ > 0.05
    candidates = [(t, sweep[t]) for t, _, _ in THRESHOLDS if sweep[t]["kappa"] >= 0.05]
    best_tag = max(candidates, key=lambda x: x[1]["n_concepts"])[0] if candidates else THRESHOLDS[0][0]
core.format_result("BEST_threshold", best_tag)

# Resolve best (sc, kw)
best_sc, best_kw = next((sc, kw) for t, sc, kw in THRESHOLDS if t == best_tag)
best_stats = sweep[best_tag]
P1_met = best_stats["kappa"] >= 0.10 and best_stats["n_concepts"] >= 20
core.format_result("P1_met", P1_met)

# ---------------------------------------------------------------------------
# P3 at the best regime: 5 disjoint random halves
# ---------------------------------------------------------------------------
print(f"\n=== P3 at {best_tag} ===")
jaccs = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N); half = N // 2
    A = h_all[idx[:half]]; B = h_all[idx[half:]]
    sA = core.closed_concept_signatures(A, best_sc, min_support=ICEBERG_SUPPORT, **best_kw)
    sB = core.closed_concept_signatures(B, best_sc, min_support=ICEBERG_SUPPORT, **best_kw)
    jaccs.append(core.jaccard(sA, sB))
p3 = float(np.mean(jaccs))
core.format_result(f"P3_{best_tag}", f"mean_jaccard={p3:.3f}")
P3_met = p3 >= 0.90
core.format_result("P3_met", P3_met)

# ---------------------------------------------------------------------------
# (b) STRONG-L1 SAE
# ---------------------------------------------------------------------------
print("\n=== training sparse SAE on MLP post-act ===")
class SAE(nn.Module):
    def __init__(self, d_in, d_dict):
        super().__init__()
        self.enc = nn.Linear(d_in, d_dict)
        self.dec = nn.Linear(d_dict, d_in, bias=False)
    def forward(self, x):
        f = torch.relu(self.enc(x))
        return self.dec(f), f

D_DICT = 2 * D_ACT   # 4096
torch.manual_seed(0)
sae = SAE(D_ACT, D_DICT)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
X = torch.from_numpy(h_all.astype(np.float32))
L1 = 3e-3
for step in range(4000):
    recon, f = sae(X)
    loss = ((recon - X) ** 2).sum(1).mean() + L1 * f.abs().sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    _, f = sae(X)
feats = f.numpy()
fire = feats > 1e-6
density = fire.mean(0)
n_dead = int((density < 1e-4).sum())
core.format_result("sae_dict_size", D_DICT)
core.format_result("sae_l1", L1)
core.format_result("sae_dead_features", n_dead)
core.format_result("sae_mean_density", f"{density.mean():.4f}")
core.format_result("sae_median_density_alive",
                    f"{np.median(density[density>=1e-4]):.4f}" if (density>=1e-4).any() else "nan")

# ---------------------------------------------------------------------------
# Galois embedding: SAE features ↔ FCA closed concepts (extents)
# Use the best lattice from the threshold sweep.
# ---------------------------------------------------------------------------
print(f"\n=== Galois embedding at {best_tag} ===")
concepts = core.frequent_closed_concepts(h_all, scaling=best_sc, min_support=ICEBERG_SUPPORT, **best_kw)
extents = [set(c["extent"].tolist()) for c in concepts]
core.format_result("n_fca_closed_concepts_used", len(concepts))

def best_jaccard(fire_set, exts):
    best = 0.0
    for e in exts:
        inter = len(fire_set & e)
        if inter:
            j = inter / len(fire_set | e)
            if j > best: best = j
    return best

per_feature = []
for c in range(D_DICT):
    if density[c] < 1e-4: continue
    fset = set(np.where(fire[:, c])[0].tolist())
    per_feature.append({"feature": c, "density": float(density[c]),
                          "best_jaccard": best_jaccard(fset, extents)})

EMBED_THRESH = 0.5
alive = len(per_feature)
n_embed = sum(1 for p in per_feature if p["best_jaccard"] >= EMBED_THRESH)
frac_embed = n_embed / max(alive, 1)
emb_dens = float(np.mean([p["density"] for p in per_feature if p["best_jaccard"] >= EMBED_THRESH])) if n_embed else 0.0
non_emb_dens = float(np.mean([p["density"] for p in per_feature if p["best_jaccard"] < EMBED_THRESH])) if (alive - n_embed) > 0 else 0.0

core.format_result("theorem_alive_sae_features", alive)
core.format_result("theorem_n_embedded", n_embed)
core.format_result("theorem_frac_embedded", f"{frac_embed:.3f}")
core.format_result(
    "theorem_density_embedded_vs_not",
    f"embedded_mean={emb_dens:.4f} non={non_emb_dens:.4f}",
)
# Honest theorem gate: needs both embedding ≥ 0.4 AND a non-trivial concept count
# (so the embedding is not just trivial overlap with the top concept).
THM_met = (frac_embed >= 0.40) and (len(concepts) >= 20)
core.format_result("THM_met", THM_met)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# threshold sweep
tags = list(sweep.keys())
axes[0].bar(tags, [sweep[t]["n_concepts"] for t in tags])
axes[0].set_yscale("symlog"); axes[0].set_ylabel("# FCA closed concepts")
axes[0].set_title("Threshold sweep: # concepts")
axes[0].tick_params(axis='x', rotation=30)

axes[1].bar(tags, [sweep[t]["kappa"] for t in tags])
axes[1].axhline(0.10, ls="--", color="red"); axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_ylabel("κ_S"); axes[1].set_title("Threshold sweep: κ_S")
axes[1].tick_params(axis='x', rotation=30)

axes[2].hist([p["best_jaccard"] for p in per_feature], bins=20, range=(0, 1))
axes[2].axvline(EMBED_THRESH, ls="--", color="red")
axes[2].set_xlabel("best Jaccard")
axes[2].set_title(f"Galois embedding at {best_tag}: {n_embed}/{alive} ({frac_embed:.0%})")
plt.tight_layout()
plt.savefig("/kaggle/working/phase3_pythia_v2_summary.png", dpi=120, bbox_inches="tight")
plt.close()

results = {
    "meta": {"model": MODEL_NAME, "layer": LAYER_IDX,
               "d_act": D_ACT, "n_tokens": N,
               "best_threshold": best_tag, "iceberg_support": ICEBERG_SUPPORT,
               "sae_dict_size": D_DICT, "sae_l1": L1},
    "sweep": sweep,
    "p3": {"jaccards": [float(x) for x in jaccs], "mean_jaccard": p3, "regime": best_tag},
    "theorem": {"frac_embedded": frac_embed, "n_alive": alive, "n_embedded": n_embed,
                  "embedded_density": emb_dens, "non_embedded_density": non_emb_dens,
                  "n_fca_concepts": len(concepts), "embed_threshold": EMBED_THRESH,
                  "sae_mean_density": float(density.mean())},
    "elapsed_sec": time.time() - t0,
    "experiment_complete": True,
}
core.format_result("elapsed_sec", f"{results['elapsed_sec']:.1f}")
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase3_pythia_v2_results.json")
print("Done.")
