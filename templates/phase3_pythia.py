"""
fca-interp phase3_pythia — Real transformer test on Pythia-70M.

The toy + MNIST experiments validated the framework's predictions under
multi-seed robustness. This notebook tests whether the same predictions hold
on a real (small) transformer's residual stream.

Pipeline:
  1. Load Pythia-70M (6 layers, d=512) from HuggingFace
  2. Tokenize ~3000 token positions from wikitext-2
  3. Extract residual-post activations from one mid-layer (layer 3)
  4. Compute κ_S at S0 (binary) and S1 (3-quantile interordinal) vs matched
     Gaussian baseline   --- tests P1 on a real transformer
  5. P3 reproducibility: disjoint random token-halves, closed-concept Jaccard
  6. Train a small SAE on the same activations (d_dict = 4·d)
  7. Galois embedding test: % of alive SAE features whose firing pattern
     matches some FCA closed concept by Jaccard ≥ 0.5

Outputs to /kaggle/working/:
  - phase3_pythia_results.json
  - phase3_pythia_summary.png
"""

import os, sys
def _find_core(name):
    for dp, _, fs in os.walk("/kaggle/input"):
        if f"{name}.py" in fs: return dp
    return None
sys.path.insert(0, _find_core("fca_core"))
import fca_core as core
import numpy as np
import torch
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

t0 = time.time()
torch.set_num_threads(4)

# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
print(f"Loading {MODEL_NAME}...")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
model.eval()
D_MODEL = model.config.hidden_size
N_LAYERS = model.config.num_hidden_layers
LAYER_IDX = 3   # mid-layer residual
core.format_result("model", MODEL_NAME)
core.format_result("d_model", D_MODEL)
core.format_result("n_layers", N_LAYERS)
core.format_result("layer_studied", LAYER_IDX)

# ---------------------------------------------------------------------------
# Text source: wikitext-2 if available, else hard-coded multi-paragraph passage
# ---------------------------------------------------------------------------
N_TOKENS_TARGET = 3000
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train",
                       trust_remote_code=True)
    # Concatenate enough non-empty lines
    chunks, total = [], 0
    for ex in ds:
        t = ex["text"].strip()
        if len(t) < 40:
            continue
        chunks.append(t)
        total += len(t)
        if total > 60_000:
            break
    raw_text = "\n\n".join(chunks)
    print(f"wikitext: {len(chunks)} chunks, {len(raw_text)} chars")
except Exception as e:
    print(f"wikitext unavailable ({e}); falling back to embedded passage")
    raw_text = (
        "The quick brown fox jumps over the lazy dog. " * 200
        + " Mathematics is the science of structure, order and relation. "
          "Formal concept analysis is a method of conceptual data analysis "
          "and knowledge representation. " * 50
    )

ids = tok(raw_text, return_tensors="pt", truncation=True,
            max_length=N_TOKENS_TARGET + 16)["input_ids"][0]
ids = ids[: N_TOKENS_TARGET + 8]
core.format_result("n_tokens_input", int(ids.numel()))

# ---------------------------------------------------------------------------
# Activation extraction: residual stream after layer LAYER_IDX
# Use a forward hook on the GPTNeoX block to capture (hidden_states,)[0]
# ---------------------------------------------------------------------------
acts = {}
def grab(_m, _i, output):
    h = output[0] if isinstance(output, tuple) else output
    acts["h"] = h.detach().float().squeeze(0)   # (T, D)

block = model.gpt_neox.layers[LAYER_IDX]
hook = block.register_forward_hook(grab)
with torch.no_grad():
    model(input_ids=ids.unsqueeze(0))
hook.remove()
h_all = acts["h"].numpy()
# Drop the first BOS-ish position and trim
h_all = h_all[1:N_TOKENS_TARGET + 1]
N = h_all.shape[0]
print(f"residual activations shape: {h_all.shape}")
core.format_result("n_tokens_used", N)

# Pythia residual stream contains negatives; for S0_zero (ReLU "fired"
# semantics) we use the abs-value > 0 mask — i.e. "active" rather than
# "positively fired". S1 quantile-based scaling is unaffected.
h_for_s0 = np.abs(h_all)

# ---------------------------------------------------------------------------
# P1: compression κ_S at S0 and S1
# ---------------------------------------------------------------------------
print("\n=== P1: compression κ_S ===")
results = {"P1": {}}
for tag, sc, kw, src in [
    ("s0_zero", "s0", {"s0_threshold": "zero"}, h_for_s0),
    ("s0_q90",  "s0", {"s0_threshold": "q90"},  h_all),
    ("s1_q3",   "s1", {"n_quantiles": 3},        h_all),
]:
    t = time.time()
    r = core.compression_ratio_kappa(src, scaling=sc, **kw, n_baseline_seeds=3)
    dt = time.time() - t
    results["P1"][tag] = r
    core.format_result(
        f"P1_{tag}",
        f"σ_C(data)={r['sigma_C_data']:.2f} "
        f"σ_C(noise)={r['sigma_C_baseline_mean']:.2f}±{r['sigma_C_baseline_std']:.2f} "
        f"κ={r['kappa']:+.3f}  (rank={r['rank']}, {dt:.1f}s)",
    )

P1_met_s1 = results["P1"]["s1_q3"]["kappa"] >= 0.10
P1_met_s0_zero = results["P1"]["s0_zero"]["kappa"] >= 0.10
P1_met = P1_met_s1 or P1_met_s0_zero
core.format_result("P1_met_s1", P1_met_s1)
core.format_result("P1_met_s0_zero", P1_met_s0_zero)
core.format_result("P1_met", P1_met)

# ---------------------------------------------------------------------------
# P3: reproducibility of frequent closed concepts across disjoint token halves
# ---------------------------------------------------------------------------
print("\n=== P3: closed-concept reproducibility ===")
P3_SUPPORT = 0.02
jaccs_s0, jaccs_s1 = [], []
for seed in range(5):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    half = N // 2
    A0, B0 = h_for_s0[idx[:half]], h_for_s0[idx[half:]]
    A1, B1 = h_all[idx[:half]], h_all[idx[half:]]
    sA0 = core.closed_concept_signatures(A0, "s0", s0_threshold="zero", min_support=P3_SUPPORT)
    sB0 = core.closed_concept_signatures(B0, "s0", s0_threshold="zero", min_support=P3_SUPPORT)
    sA1 = core.closed_concept_signatures(A1, "s1", n_quantiles=3, min_support=P3_SUPPORT)
    sB1 = core.closed_concept_signatures(B1, "s1", n_quantiles=3, min_support=P3_SUPPORT)
    jaccs_s0.append(core.jaccard(sA0, sB0))
    jaccs_s1.append(core.jaccard(sA1, sB1))

p3_s0 = float(np.mean(jaccs_s0))
p3_s1 = float(np.mean(jaccs_s1))
results["P3"] = {"s0_jaccard_mean": p3_s0, "s0_per": [float(x) for x in jaccs_s0],
                  "s1_jaccard_mean": p3_s1, "s1_per": [float(x) for x in jaccs_s1],
                  "support": P3_SUPPORT}
core.format_result("P3_s0_jaccard", f"{p3_s0:.3f}")
core.format_result("P3_s1_jaccard", f"{p3_s1:.3f}")
P3_met = (p3_s0 >= 0.90) or (p3_s1 >= 0.90)
core.format_result("P3_met", P3_met)

# ---------------------------------------------------------------------------
# Train a small SAE on the residual activations, then test Galois embedding
# ---------------------------------------------------------------------------
print("\n=== Theorem: Galois embedding (SAE features ↔ FCA concepts) ===")
import torch.nn as nn

class SAE(nn.Module):
    def __init__(self, d_in, d_dict):
        super().__init__()
        self.enc = nn.Linear(d_in, d_dict)
        self.dec = nn.Linear(d_dict, d_in, bias=False)
    def forward(self, x):
        f = torch.relu(self.enc(x))
        return self.dec(f), f

D_DICT = 4 * D_MODEL
torch.manual_seed(0)
sae = SAE(D_MODEL, D_DICT)
opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
X = torch.from_numpy(h_all.astype(np.float32))
L1 = 5e-4
print(f"  training SAE: d_dict={D_DICT}, l1={L1}")
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
core.format_result("sae_dead_features", n_dead)
core.format_result("sae_mean_density", f"{density.mean():.4f}")

# FCA closed concepts on h_for_s0 (the binary feature universe most natural
# for comparing against SAE feature firings — both are "is this active?")
concepts = core.frequent_closed_concepts(h_for_s0, scaling="s0",
                                          s0_threshold="zero", min_support=0.01)
extents = [set(c["extent"].tolist()) for c in concepts]
core.format_result("n_fca_closed_concepts", len(concepts))

def best_jaccard(fire_set, exts):
    best = 0.0
    for e in exts:
        inter = len(fire_set & e)
        if inter:
            j = inter / len(fire_set | e)
            if j > best:
                best = j
    return best

per_feature = []
for c in range(D_DICT):
    if density[c] < 1e-4:
        continue
    fset = set(np.where(fire[:, c])[0].tolist())
    bj = best_jaccard(fset, extents)
    per_feature.append({"feature": c, "density": float(density[c]), "best_jaccard": bj})

EMBED_THRESH = 0.5
alive = len(per_feature)
n_embed = sum(1 for p in per_feature if p["best_jaccard"] >= EMBED_THRESH)
frac_embed = n_embed / max(alive, 1)
embedded_density = float(np.mean([p["density"] for p in per_feature if p["best_jaccard"] >= EMBED_THRESH])) if n_embed else 0.0
non_embed_density = float(np.mean([p["density"] for p in per_feature if p["best_jaccard"] < EMBED_THRESH])) if (alive - n_embed) > 0 else 0.0

results["theorem"] = {
    "n_alive_sae_features": alive,
    "n_embedded": n_embed,
    "frac_embedded": frac_embed,
    "embedded_density": embedded_density,
    "non_embedded_density": non_embed_density,
    "embed_threshold": EMBED_THRESH,
}
core.format_result("theorem_alive_sae_features", alive)
core.format_result("theorem_n_embedded", n_embed)
core.format_result("theorem_frac_embedded", f"{frac_embed:.3f}")
core.format_result(
    "theorem_density_embedded_vs_not",
    f"embedded_mean={embedded_density:.4f} non={non_embed_density:.4f}",
)
THM_met = frac_embed >= 0.40
core.format_result("THM_met", THM_met)

# ---------------------------------------------------------------------------
# Plot summary
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
# Left: κ_S per scaling
tags = list(results["P1"].keys())
kappas = [results["P1"][t]["kappa"] for t in tags]
axes[0].bar(tags, kappas, color=["#888", "#777", "#3a7"])
axes[0].axhline(0.10, ls="--", color="red", label="P1 gate")
axes[0].axhline(0.0, color="k", lw=0.5)
axes[0].set_ylabel("κ_S (compression ratio)")
axes[0].set_title("Pythia-70M layer %d: P1" % LAYER_IDX)
axes[0].legend()

# Right: SAE embedding histogram
axes[1].hist([p["best_jaccard"] for p in per_feature], bins=20, range=(0, 1))
axes[1].axvline(EMBED_THRESH, ls="--", color="red", label=f"embed threshold={EMBED_THRESH}")
axes[1].set_xlabel("best Jaccard(SAE feature support, FCA concept extent)")
axes[1].set_ylabel("# SAE features")
axes[1].set_title(f"Galois embedding: {n_embed}/{alive} ({frac_embed:.0%})")
axes[1].legend()
plt.tight_layout()
plt.savefig("/kaggle/working/phase3_pythia_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
results["meta"] = {"model": MODEL_NAME, "layer": LAYER_IDX, "d_model": D_MODEL,
                    "n_tokens": int(N), "p3_support": P3_SUPPORT,
                    "sae_dict_size": D_DICT, "sae_l1": L1}
results["elapsed_sec"] = time.time() - t0
results["experiment_complete"] = True
core.format_result("elapsed_sec", f"{results['elapsed_sec']:.1f}")
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase3_pythia_results.json")
print("\nDone.")
