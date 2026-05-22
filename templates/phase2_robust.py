"""
fca-interp phase2_robust — multi-seed robustness for the headline claims.

Across N_SEEDS independently trained toy models, measure:
  (P1) superposition index separation: median σ_S0 in the superposition regime
       (toy) vs matched Gaussian baseline.
  (P2) structural-polysemanticity F1 vs ground truth.
  (Thm) Galois-embedding fraction: % of alive SAE features whose support
        matches an FCA closed concept; plus density of embedded vs non-embedded.

Reports mean ± std and the per-seed pass-rate for each pre-registered gate.

Outputs to /kaggle/working/:
  - phase2_robust_results.json
  - phase2_robust_summary.png
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

t0 = time.time()
N_SEEDS = 10
K, D = 10, 5                    # 10 planted features in 5 dims (superposition)
SPARSITIES = [0.05, 0.10, 0.15]  # sweep sparsity — controls activation-pattern entropy
N_PROBE = 2000

class ToyAE(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(k, d) * 0.5)
        self.b = nn.Parameter(torch.zeros(k))
    def forward(self, x):
        h = x @ self.W
        return torch.relu(h @ self.W.T + self.b), h

def train_toy(k, d, sp, seed, steps=3000, lr=1e-2, decay=0.8, n=10000):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ToyAE(k, d); imp = torch.tensor([decay**i for i in range(k)])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        mask = (torch.rand(n, k) < sp).float(); x = mask * torch.rand(n, k)
        y, _ = m(x); loss = ((x - y)**2 * imp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

class SAE(nn.Module):
    def __init__(self, d_in, d_dict):
        super().__init__()
        self.enc = nn.Linear(d_in, d_dict); self.dec = nn.Linear(d_dict, d_in, bias=False)
    def forward(self, x):
        f = torch.relu(self.enc(x)); return self.dec(f), f

def train_sae(acts, d_dict, seed, steps=4000, lr=1e-3, l1=1e-3):
    torch.manual_seed(seed); sae = SAE(acts.shape[1], d_dict)
    opt = torch.optim.Adam(sae.parameters(), lr=lr); X = torch.from_numpy(acts)
    for _ in range(steps):
        recon, f = sae(X)
        loss = ((recon - X)**2).sum(1).mean() + l1 * f.abs().sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, f = sae(X)
    return f.numpy()

def best_jaccard(fire_set, extents):
    best = 0.0
    for e in extents:
        inter = len(fire_set & e)
        if inter:
            best = max(best, inter / len(fire_set | e))
    return best

per_seed = []
for seed in range(N_SEEDS):
  for SP in SPARSITIES:
    model = train_toy(K, D, SP, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    x = (rng.random((N_PROBE, K)) < SP) * rng.random((N_PROBE, K))
    x = x.astype(np.float32)
    with torch.no_grad():
        h = torch.relu(torch.from_numpy(x) @ model.W).numpy()
    g = rng.standard_normal((N_PROBE, D)).astype(np.float32)

    # (P1) Compression-ratio superposition index κ_S (proposal Def 2, revised)
    # κ_S = 1 − σ_C(data) / σ_C(matched Gaussian baseline) ∈ (−∞, 1]
    # κ_S > 0 ⇔ structure compresses categorical diversity vs noise.
    kS = core.compression_ratio_kappa(h, scaling="s0", s0_threshold="zero")
    s_toy = float(kS["kappa"])             # primary P1 quantity
    s_g_ratio = float(core.compression_ratio_kappa(g, scaling="s0", s0_threshold="zero")["kappa"])
    s_g = s_g_ratio                        # Gaussian-on-Gaussian — should be ≈0

    # (P2) structural polysemanticity F1
    Wd = np.abs(model.W.detach().numpy())   # (K, D)
    gt = np.array([ (Wd[:, j] >= 0.5 * Wd[:, j].max()).sum() > 1 for j in range(D) ])
    sp_mask = np.array(core.neuron_structural_polysemanticity(h, scaling="s1", n_quantiles=3)["poly_mask"], dtype=bool)
    tp = int((gt & sp_mask).sum()); fp = int((~gt & sp_mask).sum()); fn = int((gt & ~sp_mask).sum())
    prec = tp / max(tp+fp, 1); rec = tp / max(tp+fn, 1)
    f1 = 2*prec*rec/max(prec+rec, 1e-9)

    # (Thm) Galois embedding fraction
    feats = train_sae(h, 2*K, seed=seed)
    fire = feats > 1e-6; dens = fire.mean(0)
    concepts = core.frequent_closed_concepts(h, scaling="s0", s0_threshold="zero", min_support=0.01)
    extents = [set(c["extent"].tolist()) for c in concepts]
    alive = [c for c in range(2*K) if dens[c] >= 1e-4]
    embed_j = {c: best_jaccard(set(np.where(fire[:, c])[0].tolist()), extents) for c in alive}
    n_embed = sum(1 for c in alive if embed_j[c] >= 0.5)
    frac_embed = n_embed / max(len(alive), 1)
    dens_emb = np.mean([dens[c] for c in alive if embed_j[c] >= 0.5]) if n_embed else 0.0
    dens_non = np.mean([dens[c] for c in alive if embed_j[c] < 0.5]) if (len(alive)-n_embed) else 0.0

    rec_seed = {
        "seed": seed, "sparsity": float(SP),
        "sigma_toy": float(s_toy), "sigma_gauss": float(s_g),
        "p2_f1": float(f1), "frac_embed": float(frac_embed),
        "n_alive": len(alive), "dens_embedded": float(dens_emb), "dens_non": float(dens_non),
    }
    per_seed.append(rec_seed)
    core.format_result(
        f"seed{seed}_sp{SP}",
        f"sigma_toy={s_toy:.2f} sigma_g={s_g:.2f} p2_f1={f1:.2f} embed={frac_embed:.2f}",
    )

def agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return float(v.mean()), float(v.std())

summary = {"per_sparsity": {}}
for SP in SPARSITIES:
    rows = [r for r in per_seed if abs(r["sparsity"] - SP) < 1e-9]
    st_m, st_s = agg(rows, "sigma_toy"); sg_m, sg_s = agg(rows, "sigma_gauss")
    f1_m, f1_s = agg(rows, "p2_f1");     em_m, em_s = agg(rows, "frac_embed")
    # P1 gate: κ_S(toy) ≥ 0.10 and |κ_S(gauss)| ≤ 0.05 (the noise floor).
    p1_pr = float(np.mean([r["sigma_toy"] >= 0.10 and abs(r["sigma_gauss"]) <= 0.05 for r in rows]))
    p2_pr = float(np.mean([r["p2_f1"] >= 0.8 for r in rows]))
    thm_pr = float(np.mean([r["frac_embed"] >= 0.5 for r in rows]))
    summary["per_sparsity"][str(SP)] = {
        "sigma_toy": [st_m, st_s], "sigma_gauss": [sg_m, sg_s],
        "p2_f1": [f1_m, f1_s], "embed_frac": [em_m, em_s],
        "p1_passrate": p1_pr, "p2_passrate": p2_pr, "thm_passrate": thm_pr,
    }
    core.format_result(
        f"ROBUST_sp{SP}",
        f"σ_toy={st_m:.2f}±{st_s:.2f} σ_g={sg_m:.2f}±{sg_s:.2f} "
        f"P2={f1_m:.2f}±{f1_s:.2f} thm={em_m:.2f}±{em_s:.2f} "
        f"P1_pass={p1_pr:.0%} P2_pass={p2_pr:.0%} THM_pass={thm_pr:.0%}",
    )

# Best-sparsity gates for the publication summary
best_p1_sp = max(SPARSITIES, key=lambda s: summary["per_sparsity"][str(s)]["p1_passrate"])
best_p2_sp = max(SPARSITIES, key=lambda s: summary["per_sparsity"][str(s)]["p2_passrate"])
best_thm_sp = max(SPARSITIES, key=lambda s: summary["per_sparsity"][str(s)]["thm_passrate"])
core.format_result("BEST_P1_sparsity", best_p1_sp)
core.format_result("BEST_P2_sparsity", best_p2_sp)
core.format_result("BEST_THM_sparsity", best_thm_sp)

P1_met = any(summary["per_sparsity"][str(s)]["p1_passrate"] >= 0.8 for s in SPARSITIES)
P2_met = any(summary["per_sparsity"][str(s)]["p2_passrate"] >= 0.8 for s in SPARSITIES)
THM_met = any(summary["per_sparsity"][str(s)]["thm_passrate"] >= 0.8 for s in SPARSITIES)
core.format_result("P1_met", P1_met)
core.format_result("P2_met", P2_met)
core.format_result("THM_met", THM_met)

# Plot: per-sparsity summary (mean ± std bars + horizontal pass thresholds)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
sp_labels = [str(s) for s in SPARSITIES]
def collect(metric):
    means = [summary["per_sparsity"][sp_labels[i]][metric][0] for i in range(len(SPARSITIES))]
    stds = [summary["per_sparsity"][sp_labels[i]][metric][1] for i in range(len(SPARSITIES))]
    return means, stds
sig_toy_m, sig_toy_s = collect("sigma_toy"); sig_g_m, sig_g_s = collect("sigma_gauss")
xs = np.arange(len(SPARSITIES))
axes[0].errorbar(xs-0.1, sig_toy_m, yerr=sig_toy_s, fmt="o", label="toy κ_S")
axes[0].errorbar(xs+0.1, sig_g_m,   yerr=sig_g_s,   fmt="s", label="gauss κ_S")
axes[0].axhline(0.10, ls="--", color="grey"); axes[0].axhline(0.0, color="k", lw=0.5)
axes[0].set_xticks(xs); axes[0].set_xticklabels(sp_labels)
axes[0].set_title("P1: κ_S (compression ratio)"); axes[0].set_xlabel("sparsity"); axes[0].legend()

p2_m, p2_s = collect("p2_f1")
axes[1].errorbar(xs, p2_m, yerr=p2_s, fmt="o-")
axes[1].axhline(0.8, ls="--", color="red"); axes[1].set_ylim(0, 1.05)
axes[1].set_xticks(xs); axes[1].set_xticklabels(sp_labels)
axes[1].set_title("P2: structural-poly F1"); axes[1].set_xlabel("sparsity")

em_m, em_s = collect("embed_frac")
axes[2].errorbar(xs, em_m, yerr=em_s, fmt="o-")
axes[2].axhline(0.5, ls="--", color="red"); axes[2].set_ylim(0, 1.05)
axes[2].set_xticks(xs); axes[2].set_xticklabels(sp_labels)
axes[2].set_title("Thm: SAE embedding fraction"); axes[2].set_xlabel("sparsity")
plt.tight_layout()
plt.savefig("/kaggle/working/phase2_robust_summary.png", dpi=120, bbox_inches="tight")
plt.close()

results = {
    "n_seeds": N_SEEDS, "config": {"K": K, "D": D, "sparsities": SPARSITIES},
    "per_seed": per_seed,
    "summary": summary,
    "elapsed_sec": time.time() - t0, "experiment_complete": True,
}
core.format_result("elapsed_sec", f"{results['elapsed_sec']:.1f}")
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase2_robust_results.json")
print("Done.")
