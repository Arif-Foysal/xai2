"""
fca-interp phase3_track_b_v2 — final transformer test.

v1 showed: at 899 kept SAE features × 3000 tokens, every token has a unique
firing signature → 0 closed concepts at any iceberg threshold. This notebook
tests two principled remedies that don't require full-context pattern repetition:

  (A) ATTRIBUTE IMPLICATIONS: for each kept SAE feature m, compute its closure
      {m}'' = features that ALWAYS co-fire with m. This is a lightweight,
      meaningful structure (implications) that exists even when full
      firing-patterns don't repeat.

  (B) TOP-K SUBSET LATTICE: pick the K most-firing kept features and build
      the full closed-concept lattice on the (N × K) binary context. With
      K=32 and N=3000, frequent patterns DO repeat → non-trivial lattice.

For each, we report counts, P3 reproducibility, and a κ_S-style comparison
against a Bernoulli null with matched per-feature densities.
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

# ----- tokens -----
N_TOKENS_TARGET = 3000
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    chunks, total = [], 0
    for ex in ds:
        t_ = ex["text"].strip()
        if len(t_) < 40: continue
        chunks.append(t_); total += len(t_)
        if total > 60_000: break
    raw_text = "\n\n".join(chunks)
except Exception:
    raw_text = "Mathematics structure order relation. " * 5000
ids = tok(raw_text, return_tensors="pt", truncation=True, max_length=N_TOKENS_TARGET + 16)["input_ids"][0]
ids = ids[: N_TOKENS_TARGET + 8]

# ----- MLP post-act -----
acts = {}
def grab(_m, _i, out):
    acts["h"] = out.detach().float().squeeze(0)
hook = model.gpt_neox.layers[LAYER_IDX].mlp.act.register_forward_hook(grab)
with torch.no_grad():
    model(input_ids=ids.unsqueeze(0))
hook.remove()
h_all = acts["h"].numpy()[1:N_TOKENS_TARGET + 1]
N, D_ACT = h_all.shape
core.format_result("n_tokens_used", N)
core.format_result("d_activation", D_ACT)

# ----- Sparse SAE -----
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
with torch.no_grad():
    _, f = sae(X)
feats = f.numpy()
fire = feats > 1e-6
density = fire.mean(0)

DENSITY_MIN, DENSITY_MAX = 0.01, 0.5
keep = (density >= DENSITY_MIN) & (density <= DENSITY_MAX)
kept_idx = np.where(keep)[0]
fire_kept = fire[:, kept_idx]
n_kept = len(kept_idx)
core.format_result("sae_n_kept_features", n_kept)
core.format_result(
    "sae_kept_density",
    f"min={density[kept_idx].min():.3f} median={np.median(density[kept_idx]):.3f} max={density[kept_idx].max():.3f}",
)

# ===========================================================================
# (A) ATTRIBUTE IMPLICATIONS
# For each kept feature m, closure({m}) = features that fire on every object
# where m fires. Reports # non-trivial implications, and reproducibility.
# ===========================================================================
print("\n=== (A) Attribute-implication closures ===")

def attribute_closures_binary(I_bin):
    """For each column m, returns the closure mask {m}'' as a bool ndarray."""
    N, M = I_bin.shape
    closures = []
    for m in range(M):
        ext = I_bin[:, m]
        if not ext.any():
            closures.append(np.zeros(M, dtype=bool))
            continue
        clo = I_bin[ext].all(axis=0)
        closures.append(clo)
    return closures

closures_data = attribute_closures_binary(fire_kept)
# Non-trivial implication: closure({m}) has > 1 feature (m → at least 1 other)
implication_counts = [int(c.sum()) - 1 for c in closures_data]   # -1 for m itself
n_nontrivial = sum(1 for x in implication_counts if x >= 1)
implication_sigs = {frozenset(np.where(c)[0].tolist()) for c in closures_data if c.sum() >= 1}
core.format_result(
    "trackB_A_implications",
    f"kept_features={n_kept} non_trivial_implications={n_nontrivial} unique_closure_sigs={len(implication_sigs)}",
)
core.format_result(
    "trackB_A_implication_size_stats",
    f"mean={np.mean(implication_counts):.2f} max={max(implication_counts)} (extra features per closure)",
)

# Bernoulli null: same per-feature density, independent
rng = np.random.default_rng(0)
n_nontrivial_bern_list, sigs_bern_list = [], []
for s in range(3):
    rng = np.random.default_rng(s + 100)
    bern = rng.random(fire_kept.shape) < density[kept_idx][None, :]
    closures_bern = attribute_closures_binary(bern.astype(bool))
    impl_bern = [int(c.sum()) - 1 for c in closures_bern]
    n_nontrivial_bern_list.append(sum(1 for x in impl_bern if x >= 1))
    sigs_bern_list.append({frozenset(np.where(c)[0].tolist()) for c in closures_bern if c.sum() >= 1})
n_nontrivial_bern = float(np.mean(n_nontrivial_bern_list))
n_unique_bern = float(np.mean([len(s) for s in sigs_bern_list]))
core.format_result(
    "trackB_A_bernoulli",
    f"non_trivial_implications={int(n_nontrivial_bern)} unique_closures={int(n_unique_bern)}",
)

# Reproducibility of attribute-closure SIGNATURES across disjoint halves
jaccs_A = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N); half = N // 2
    cA = attribute_closures_binary(fire_kept[idx[:half]])
    cB = attribute_closures_binary(fire_kept[idx[half:]])
    sA = {frozenset(np.where(c)[0].tolist()) for c in cA if c.sum() >= 1}
    sB = {frozenset(np.where(c)[0].tolist()) for c in cB if c.sum() >= 1}
    jaccs_A.append(core.jaccard(sA, sB))
p3_A = float(np.mean(jaccs_A))
core.format_result(
    "trackB_A_P3_jaccard",
    f"mean={p3_A:.3f} per={[round(x,2) for x in jaccs_A]}",
)

# ===========================================================================
# (B) TOP-K SUBSET LATTICE
# ===========================================================================
print("\n=== (B) Top-K closed-concept lattice ===")
K = 32
top_idx = np.argsort(-density[kept_idx])[:K]
fire_topK = fire_kept[:, top_idx]
core.format_result("trackB_B_top_K_density_stats",
                    f"min={density[kept_idx][top_idx].min():.3f} "
                    f"median={np.median(density[kept_idx][top_idx]):.3f} "
                    f"max={density[kept_idx][top_idx].max():.3f}")

def closed_concepts_binary(I_bin, min_support):
    N = I_bin.shape[0]
    mc = int(np.ceil(min_support * N)) if 0 < min_support < 1 else int(min_support)
    seen = {}
    Io, _ = core.clarify_objects(I_bin)
    for g in range(Io.shape[0]):
        row = Io[g]
        ext_clar = np.all(Io >= row, axis=1) if row.any() else np.ones(Io.shape[0], dtype=bool)
        intent = Io[ext_clar].all(axis=0) if ext_clar.any() else np.ones(Io.shape[1], dtype=bool)
        key = intent.tobytes()
        if key in seen:
            continue
        support = int(np.all(I_bin >= intent[None, :], axis=1).sum())
        if support < mc:
            continue
        seen[key] = {"intent": intent, "support": support}
    return list(seen.values())

ICEBERG_B = 0.01
concepts_B = closed_concepts_binary(fire_topK, min_support=ICEBERG_B)
core.format_result(
    "trackB_B_n_closed_concepts",
    f"{len(concepts_B)} (K={K}, support ≥ {int(np.ceil(ICEBERG_B*N))} tokens)",
)
# top concepts
top_concepts = sorted(concepts_B, key=lambda c: -c["support"])[:5]
for i, c in enumerate(top_concepts):
    fs = np.where(c["intent"])[0]
    core.format_result(
        f"trackB_B_top_concept_{i}",
        f"support={c['support']} |intent|={len(fs)} features={fs.tolist()[:10]}",
    )

# Bernoulli null
bern_concept_counts = []
for s in range(3):
    rng = np.random.default_rng(s + 200)
    bern = rng.random(fire_topK.shape) < density[kept_idx][top_idx][None, :]
    bern_concept_counts.append(len(closed_concepts_binary(bern.astype(bool), ICEBERG_B)))
ratio_B = len(concepts_B) / max(np.mean(bern_concept_counts), 1)
core.format_result(
    "trackB_B_concepts_vs_bernoulli",
    f"data={len(concepts_B)} bernoulli={int(np.mean(bern_concept_counts))} ratio={ratio_B:.2f}",
)

# P3 on top-K lattice
jaccs_B = []
for seed in range(5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N); half = N // 2
    cA = closed_concepts_binary(fire_topK[idx[:half]], ICEBERG_B)
    cB = closed_concepts_binary(fire_topK[idx[half:]], ICEBERG_B)
    sA = {frozenset(np.where(c["intent"])[0].tolist()) for c in cA}
    sB = {frozenset(np.where(c["intent"])[0].tolist()) for c in cB}
    jaccs_B.append(core.jaccard(sA, sB))
p3_B = float(np.mean(jaccs_B))
core.format_result("trackB_B_P3_jaccard",
                    f"mean={p3_B:.3f} per={[round(x,2) for x in jaccs_B]}")

# ===========================================================================
# Headline gates
# ===========================================================================
# (A) Attribute implications: structure exists iff non-trivial-implication
#     count > Bernoulli baseline by a factor ≥ 2, AND reproducibility ≥ 0.5
A_ok = (n_nontrivial >= 2 * n_nontrivial_bern) and (p3_A >= 0.5)
# (B) Top-K lattice: at least 5 concepts and ratio ≥ 2 over Bernoulli, P3 ≥ 0.5
B_ok = (len(concepts_B) >= 5) and (ratio_B >= 2.0) and (p3_B >= 0.5)
P1_met = A_ok or B_ok
P3_met = (p3_A >= 0.5) or (p3_B >= 0.5)
core.format_result("trackB_A_ok", A_ok)
core.format_result("trackB_B_ok", B_ok)
core.format_result("P1_met", P1_met)
core.format_result("P3_met", P3_met)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar(["A: data", "A: bernoulli"], [n_nontrivial, n_nontrivial_bern])
axes[0].set_title("(A) Non-trivial implications")
axes[1].bar(["B: data", "B: bernoulli"], [len(concepts_B), np.mean(bern_concept_counts)])
axes[1].set_title(f"(B) Top-{K} closed concepts")
axes[2].bar(["A P3", "B P3"], [p3_A, p3_B]); axes[2].axhline(0.5, ls="--", color="red")
axes[2].set_ylim(0, 1.05); axes[2].set_title("Reproducibility")
plt.tight_layout()
plt.savefig("/kaggle/working/phase3_track_b_v2_summary.png", dpi=120, bbox_inches="tight")
plt.close()

results = {
    "meta": {"model": MODEL_NAME, "layer": LAYER_IDX, "n_tokens": N,
               "sae_dict": D_DICT, "sae_l1": L1, "n_kept": n_kept,
               "K_top": K, "iceberg_B": ICEBERG_B},
    "A_implications": {"n_kept": n_kept,
                         "non_trivial_implications": n_nontrivial,
                         "non_trivial_implications_bern": n_nontrivial_bern,
                         "unique_closures": len(implication_sigs),
                         "unique_closures_bern": n_unique_bern,
                         "p3_jaccard": p3_A, "ok": A_ok},
    "B_topK_lattice": {"K": K, "n_concepts": len(concepts_B),
                         "n_concepts_bern_mean": float(np.mean(bern_concept_counts)),
                         "concept_ratio": float(ratio_B),
                         "p3_jaccard": p3_B, "ok": B_ok,
                         "top_concept_supports": [int(c["support"]) for c in top_concepts]},
    "P1_met": P1_met, "P3_met": P3_met,
    "elapsed_sec": time.time() - t0, "experiment_complete": True,
}
core.format_result("elapsed_sec", f"{results['elapsed_sec']:.1f}")
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase3_track_b_v2_results.json")
print("Done.")
