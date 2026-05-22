"""
fca-interp phase2_p1rescue — diagnose P1 (compression κ_S) regime sensitivity.

The v3 robustness sweep at S0 found κ_S(trained toy) ≈ 0.02–0.05 (target 0.10)
— too small to be a robust gate. Hypothesis: training drives neurons to fire
on most inputs (saturation), so the binary fire/no-fire context collapses.

This notebook tests three remedies in one pass across 10 seeds × 3 sparsities:
  (a) κ_S at S1 (3-quantile interordinal scaling) on TRAINED activations
  (b) κ_S at S0 on UNTRAINED (random-projection) activations — confirms the
      mechanism is training-driven, not architectural
  (c) κ_S at S0 with quantile-threshold (q90) on trained activations

Reports per-(sparsity, treatment) means/stds + pass-rate at κ ≥ 0.10.
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

t0 = time.time()
N_SEEDS = 10
K, D = 10, 5
SPARSITIES = [0.05, 0.10, 0.15]
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

def random_proj(k, d, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((k, d)).astype(np.float32) * 0.5

def probe(W_kd, k, sp, seed):
    rng = np.random.default_rng(1000 + seed)
    x = (rng.random((N_PROBE, k)) < sp) * rng.random((N_PROBE, k))
    return np.maximum(0, x.astype(np.float32) @ W_kd)

TREATMENTS = [
    ("trained_s0_zero", "s0", {"s0_threshold": "zero"}, True),
    ("trained_s0_q90",  "s0", {"s0_threshold": "q90"},  True),
    ("trained_s1_q3",   "s1", {"n_quantiles": 3},        True),
    ("random_s0_zero",  "s0", {"s0_threshold": "zero"}, False),
    ("random_s1_q3",    "s1", {"n_quantiles": 3},        False),
]

rows = []
for seed in range(N_SEEDS):
    for SP in SPARSITIES:
        model = train_toy(K, D, SP, seed=seed)
        W_trained = model.W.detach().numpy()
        W_random = random_proj(K, D, seed=seed + 7777)
        for tag, sc, kw, trained in TREATMENTS:
            W = W_trained if trained else W_random
            h = probe(W, K, SP, seed)
            r = core.compression_ratio_kappa(h, scaling=sc, **kw)
            rows.append({"seed": seed, "sparsity": SP, "treatment": tag,
                         "kappa": r["kappa"],
                         "sigma_C_data": r["sigma_C_data"],
                         "sigma_C_base": r["sigma_C_baseline_mean"]})
        core.format_result(
            f"s{seed}_sp{SP}",
            " ".join(
                f"{t[0]}=κ{rows[-len(TREATMENTS)+i]['kappa']:+.2f}"
                for i, t in enumerate(TREATMENTS)
            ),
        )

summary = {}
for tag, _, _, _ in TREATMENTS:
    for SP in SPARSITIES:
        sub = [r for r in rows if r["treatment"] == tag and abs(r["sparsity"] - SP) < 1e-9]
        ks = np.array([r["kappa"] for r in sub])
        passrate = float((ks >= 0.10).mean())
        key = f"{tag}_sp{SP}"
        summary[key] = {"mean": float(ks.mean()), "std": float(ks.std()),
                        "passrate": passrate, "n": len(sub)}
        core.format_result(key, f"κ={ks.mean():+.3f}±{ks.std():.3f} pass(κ≥0.10)={passrate:.0%}")

# Best treatment: any per-sparsity passrate ≥ 0.8
best_pass = {}
for tag, _, _, _ in TREATMENTS:
    best_pr = max(summary[f"{tag}_sp{SP}"]["passrate"] for SP in SPARSITIES)
    best_pass[tag] = best_pr
    core.format_result(f"BEST_{tag}_passrate", f"{best_pr:.0%}")

P1_met = any(p >= 0.8 for p in best_pass.values())
core.format_result("P1_met", P1_met)
core.format_result("BEST_treatment", max(best_pass, key=best_pass.get))

results = {"per_run": rows, "summary": summary, "best_passrate_per_treatment": best_pass,
           "P1_met": P1_met, "elapsed_sec": time.time() - t0,
           "experiment_complete": True}
core.format_result("elapsed_sec", f"{results['elapsed_sec']:.1f}")
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase2_p1rescue_results.json")
print("Done.")
