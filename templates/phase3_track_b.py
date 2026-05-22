"""
fca-interp phase3_track_b — Track B: FCA on SAE feature firings (proposal §3.4).

Track A (FCA on raw transformer activations) was degenerate: with d_act=2048
binary attributes and only n=3000 tokens, every token has a near-unique firing
signature → 0 closed concepts at any iceberg threshold. Track B addresses this
by using SAE features themselves as the binary attributes — SAEs are designed
to be sparse, so co-firing patterns DO repeat across tokens.

Pipeline:
  1. MLP post-act of Pythia-70M layer 3 (as v2)
  2. Train a sparse SAE with L1=1e-2 (target per-feature density ≤ 0.05)
  3. Drop dead / always-on features; keep features with 0.005 ≤ density ≤ 0.5
  4. Build binary context: (tokens × kept SAE features), feature fires iff
     activation > 0. This is the Track B context.
  5. Closed concepts on this context = co-firing groups of SAE features.
  6. P1: closed-concept count and κ_S vs matched-density Bernoulli baseline
     (independent SAE-firings null model, the right baseline here, not
     Gaussian).
  7. P3: closed-concept Jaccard on disjoint token halves.
  8. Concept "interpretability" sanity check: report a few of the largest
     closed concepts as (feature-index-set, support).
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
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()
D_MODEL = model.config.hidden_size
LAYER_IDX = 3
core.format_result("model", MODEL_NAME)
core.format_result("layer_studied", LAYER_IDX)

# ---------------------------------------------------------------------------
# Tokenize
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
    print(f"wikitext unavailable: {e}")
    raw_text = "Mathematics structure order relation. " * 5000
ids = tok(raw_text, return_tensors="pt", truncation=True, max_length=N_TOKENS_TARGET + 16)["input_ids"][0]
ids = ids[: N_TOKENS_TARGET + 8]

# ---------------------------------------------------------------------------
# MLP post-activation extraction
# ---------------------------------------------------------------------------
acts = {}
def grab(_m, _i, output):
    acts["h"] = output.detach().float().squeeze(0)
hook = model.gpt_neox.layers[LAYER_IDX].mlp.act.register_forward_hook(grab)
with torch.no_grad():
    model(input_ids=ids.unsqueeze(0))
hook.remove()
h_all = acts["h"].numpy()[1:N_TOKENS_TARGET + 1]
N, D_ACT = h_all.shape
core.format_result("n_tokens_used", N)
core.format_result("d_activation", D_ACT)

# ---------------------------------------------------------------------------
# Train SPARSE SAE (L1=1e-2)
# ---------------------------------------------------------------------------
class SAE(nn.Module):
    def __init__(self, d_in, d_dict):
        super().__init__()
        self.enc = nn.Linear(d_in, d_dict)
        self.dec = nn.Linear(d_dict, d_in, bias=False)
    def forward(self, x):
        f = torch.relu(self.enc(x))
        return self.dec(f), f

D_DICT = 2 * D_ACT
L1 = 1e-2
torch.manual_seed(0)
sae = SAE(D_ACT, D_DICT)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
X = torch.from_numpy(h_all.astype(np.float32))
print(f"training SAE: d_dict={D_DICT} L1={L1}")
for step in range(4000):
    recon, f = sae(X)
    loss = ((recon - X) ** 2).sum(1).mean() + L1 * f.abs().sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 999:
        with torch.no_grad():
            _, f0 = sae(X)
        d0 = (f0 > 1e-6).float().mean().item()
        recon_mse = ((recon - X) ** 2).mean().item()
        print(f"  step {step+1}: mean density {d0:.4f}  recon_mse {recon_mse:.4f}")
with torch.no_grad():
    _, f = sae(X)
feats = f.numpy()
fire = feats > 1e-6
density = fire.mean(0)

core.format_result("sae_dict_size", D_DICT)
core.format_result("sae_l1", L1)
core.format_result("sae_mean_density_all", f"{density.mean():.4f}")
core.format_result("sae_median_density_all", f"{np.median(density):.4f}")

# ---------------------------------------------------------------------------
# Keep features in the "useful" density band: not dead, not always-on
# ---------------------------------------------------------------------------
DENSITY_MIN = 0.005   # 15 tokens
DENSITY_MAX = 0.50
keep = (density >= DENSITY_MIN) & (density <= DENSITY_MAX)
fire_kept = fire[:, keep]
n_kept = int(keep.sum())
core.format_result("sae_n_kept_features", n_kept)
core.format_result(
    "sae_kept_density_stats",
    f"min={density[keep].min() if n_kept else 0:.3f} "
    f"median={np.median(density[keep]) if n_kept else 0:.3f} "
    f"max={density[keep].max() if n_kept else 0:.3f}",
)

# ---------------------------------------------------------------------------
# Track B: FCA closed concepts directly on the (N, n_kept) binary matrix
# ---------------------------------------------------------------------------
def closed_concepts_binary(I_bin, min_support=0.005):
    """Frequent closed concepts on an already-binary context I_bin (N, M)."""
    N = I_bin.shape[0]
    mc = int(np.ceil(min_support * N)) if 0 < min_support < 1 else int(min_support)
    seen = {}
    # use clarify to drop duplicate rows for speed
    Io, _ = core.clarify_objects(I_bin)
    for g in range(Io.shape[0]):
        row = Io[g]
        ext_clar = np.all(Io >= row, axis=1) if row.any() else np.ones(Io.shape[0], dtype=bool)
        intent = Io[ext_clar].all(axis=0) if ext_clar.any() else np.ones(Io.shape[1], dtype=bool)
        key = intent.tobytes()
        if key in seen:
            continue
        # support in original objects: row repeats accounted for via clarification
        # Compute exact support in I_bin by recounting:
        support = int(np.all(I_bin >= intent[None, :], axis=1).sum())
        if support < mc:
            continue
        seen[key] = {"intent": intent, "support": support}
    return list(seen.values())


def signatures_from_concepts(concepts):
    return {frozenset(np.where(c["intent"])[0].tolist()) for c in concepts}


print("\n=== Track B: closed concepts on SAE-firing context ===")
ICEBERG = 0.005
t = time.time()
concepts_b = closed_concepts_binary(fire_kept, min_support=ICEBERG)
dt = time.time() - t
core.format_result(
    "trackB_n_closed_concepts",
    f"{len(concepts_b)} (built in {dt:.1f}s, support ≥ {int(np.ceil(ICEBERG*N))} tokens)",
)

# Top concepts: report a few
top_concepts = sorted(concepts_b, key=lambda c: -c["support"])[:5]
for i, c in enumerate(top_concepts):
    fs = np.where(c["intent"])[0]
    core.format_result(
        f"trackB_top_concept_{i}",
        f"support={c['support']} |intent|={len(fs)} features={list(fs[:8])}{'...' if len(fs)>8 else ''}",
    )

# ---------------------------------------------------------------------------
# κ_S vs MATCHED-DENSITY BERNOULLI baseline (correct null model for binary SAE firings)
# ---------------------------------------------------------------------------
print("\n=== Bernoulli null model (matched per-feature density) ===")
rng = np.random.default_rng(0)
bern_concept_counts = []
n_obj_clar_data = core.clarify_objects(fire_kept)[0].shape[0]
n_obj_clar_baselines = []
for s in range(3):
    rng = np.random.default_rng(s + 100)
    bern = rng.random(fire_kept.shape) < density[keep][None, :]
    bern_n_obj_clar = core.clarify_objects(bern.astype(bool))[0].shape[0]
    n_obj_clar_baselines.append(bern_n_obj_clar)
    bern_concept_counts.append(len(closed_concepts_binary(bern, min_support=ICEBERG)))
sigma_C_data_b = n_obj_clar_data / max(np.linalg.matrix_rank(fire_kept.astype(float)), 1)
sigma_C_base_b = float(np.mean(n_obj_clar_baselines)) / max(np.linalg.matrix_rank(fire_kept.astype(float)), 1)
kappa_b = 1 - sigma_C_data_b / max(sigma_C_base_b, 1e-9)
core.format_result(
    "trackB_n_obj_clar",
    f"data={n_obj_clar_data} bernoulli={int(np.mean(n_obj_clar_baselines))}",
)
core.format_result("trackB_n_concepts_vs_bernoulli",
                    f"data={len(concepts_b)} bernoulli={int(np.mean(bern_concept_counts))}")
core.format_result("trackB_kappa", f"{kappa_b:+.3f}")

# Concept-count ratio — sharper indicator than κ_S in Track B
concept_ratio = len(concepts_b) / max(np.mean(bern_concept_counts), 1)
core.format_result("trackB_concept_ratio_data_over_bern", f"{concept_ratio:.2f}")

P1_met = (len(concepts_b) >= 20) and (concept_ratio >= 2.0 or kappa_b >= 0.10)
core.format_result("P1_met", P1_met)

# ---------------------------------------------------------------------------
# P3: split-half closed-concept Jaccard
# ---------------------------------------------------------------------------
print("\n=== P3: reproducibility ===")
jaccs = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N); half = N // 2
    cA = closed_concepts_binary(fire_kept[idx[:half]], min_support=ICEBERG)
    cB = closed_concepts_binary(fire_kept[idx[half:]], min_support=ICEBERG)
    j = core.jaccard(signatures_from_concepts(cA), signatures_from_concepts(cB))
    jaccs.append(j)
p3 = float(np.mean(jaccs))
core.format_result("P3_trackB", f"mean_jaccard={p3:.3f} per={[round(x,2) for x in jaccs]}")
P3_met = p3 >= 0.90
core.format_result("P3_met", P3_met)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(density, bins=40); axes[0].axvline(DENSITY_MIN, ls="--", color="red")
axes[0].axvline(DENSITY_MAX, ls="--", color="red")
axes[0].set_xlabel("SAE feature density"); axes[0].set_ylabel("# features")
axes[0].set_title(f"SAE density (kept {n_kept}/{D_DICT})")

axes[1].bar(["data", "bernoulli"], [len(concepts_b), np.mean(bern_concept_counts)])
axes[1].set_ylabel("# closed concepts")
axes[1].set_title(f"Track B closed concepts (κ={kappa_b:+.2f})")

axes[2].bar(range(len(jaccs)), jaccs)
axes[2].axhline(0.90, ls="--", color="red")
axes[2].set_ylim(0, 1.05)
axes[2].set_xlabel("split seed"); axes[2].set_ylabel("Jaccard")
axes[2].set_title(f"P3: mean={p3:.2f}")
plt.tight_layout()
plt.savefig("/kaggle/working/phase3_track_b_summary.png", dpi=120, bbox_inches="tight")
plt.close()

results = {
    "meta": {"model": MODEL_NAME, "layer": LAYER_IDX, "d_act": D_ACT,
               "n_tokens": N, "sae_dict": D_DICT, "sae_l1": L1,
               "density_band": [DENSITY_MIN, DENSITY_MAX],
               "iceberg_support": ICEBERG},
    "sae": {"mean_density_all": float(density.mean()),
              "median_density_all": float(np.median(density)),
              "n_kept_features": n_kept},
    "trackB": {"n_closed_concepts": len(concepts_b),
                "n_concepts_bernoulli_mean": float(np.mean(bern_concept_counts)),
                "concept_ratio": float(concept_ratio),
                "kappa": float(kappa_b),
                "top_concept_supports": [int(c["support"]) for c in top_concepts]},
    "p3": {"per_seed_jaccards": [float(x) for x in jaccs],
            "mean_jaccard": p3},
    "elapsed_sec": time.time() - t0,
    "experiment_complete": True,
    "P1_met": P1_met, "P3_met": P3_met,
}
core.format_result("elapsed_sec", f"{results['elapsed_sec']:.1f}")
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase3_track_b_results.json")
print("Done.")
