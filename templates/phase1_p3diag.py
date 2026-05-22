"""
fca-interp phase1_p3diag — diagnose P3 reproducibility on the REAL trained
toy model. Fast (~15s). Prints the actual signature sets, their symmetric
differences, and four candidate reproducibility invariants so we can choose
the correct one rather than guess locally without torch.

Outputs to /kaggle/working/:
  - phase1_p3diag_results.json
"""

import os, sys
def _find_core(name):
    for dp,_,fs in os.walk("/kaggle/input"):
        if f"{name}.py" in fs: return dp
    return None
sys.path.insert(0, _find_core("fca_core"))
import fca_core as core
import numpy as np
import torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)

class ToyAE(nn.Module):
    def __init__(self, k, d):
        super().__init__()
        self.W = nn.Parameter(torch.randn(k, d) * 0.5)
        self.b = nn.Parameter(torch.zeros(k))
    def forward(self, x):
        h = x @ self.W
        return torch.relu(h @ self.W.T + self.b), h

def train(k, d, sp, steps=2000, lr=1e-2, decay=0.7, seed=0, n=10000):
    torch.manual_seed(seed); np.random.seed(seed)
    m = ToyAE(k, d); imp = torch.tensor([decay**i for i in range(k)])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        mask = (torch.rand(n, k) < sp).float(); x = mask * torch.rand(n, k)
        y, _ = m(x); loss = ((x-y)**2 * imp).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m

K, D, SP = 10, 5, 0.1
model = train(K, D, SP, seed=5)
rng = np.random.default_rng(42)
mask = (rng.random((3200, K)) < SP).astype(np.float32)
x = mask * rng.random((3200, K)).astype(np.float32)
with torch.no_grad():
    h = torch.relu(torch.from_numpy(x) @ model.W).numpy()

print("frac nonzero per neuron:", np.round((h > 1e-12).mean(0), 3))
pats = {tuple((h[i] > 1e-12).tolist()) for i in range(len(h))}
print("distinct fire patterns:", len(pats))
core.format_result("diag_distinct_patterns", len(pats))
core.format_result("diag_fire_fractions", np.round((h > 1e-12).mean(0), 3).tolist())

def split(seed):
    r = np.random.default_rng(seed); idx = r.permutation(3200)
    return h[idx[:1600]], h[idx[1600:]]

results = {"metrics": {}}

# Candidate invariants
def closed_intents(act, ms, sc="s0", **kw):
    if sc == "s0":
        I, labels = core.interordinal_context(act, scaling="s0", **kw)
    else:
        I, labels = core.interordinal_context(act, scaling=sc, **kw)
    n = I.shape[0]; mc = int(np.ceil(ms*n)) if 0 < ms < 1 else int(ms)
    Ic, _, cm = core.clarify_context(I, labels)
    clab = {}
    for raw, lab in enumerate(labels): clab.setdefault(int(cm[raw]), lab)
    Io, rmap = core.clarify_objects(Ic); mult = np.bincount(rmap, minlength=Io.shape[0])
    sigs = set()
    for g in range(Io.shape[0]):
        row = Io[g]; ext = np.all(Io >= row, axis=1); sup = int(mult[ext].sum())
        if sup < mc: continue
        intent = np.all(Io[ext], axis=0)
        sigs.add(frozenset(clab[int(c)] for c in np.where(intent)[0]))
    return sigs

CANDIDATES = {
    "join_irred_s0_ms0":   lambda A: core.join_irreducible_signatures(A, "s0", s0_threshold="zero", min_support=0.0),
    "join_irred_s0_ms02":  lambda A: core.join_irreducible_signatures(A, "s0", s0_threshold="zero", min_support=0.02),
    "attr_closure_s0":     lambda A: core.attribute_closures(A, "s0", s0_threshold="zero"),
    "attr_closure_s1":     lambda A: core.attribute_closures(A, "s1", n_quantiles=3),
    "closed_intent_s0_ms02": lambda A: closed_intents(A, 0.02, "s0", s0_threshold="zero"),
    "closed_intent_s0_ms05": lambda A: closed_intents(A, 0.05, "s0", s0_threshold="zero"),
}

for name, fn in CANDIDATES.items():
    jaccs = []
    for seed in range(5):
        A, B = split(seed)
        jaccs.append(core.jaccard(fn(A), fn(B)))
    mean_j = float(np.mean(jaccs))
    results["metrics"][name] = {"mean_jaccard": mean_j, "per": [float(x) for x in jaccs]}
    core.format_result(f"diag_{name}", f"mean={mean_j:.3f} per={[round(x,2) for x in jaccs]}")

# Detailed look at the failing one: join_irred sets for seed 0
A, B = split(0)
sA = core.join_irreducible_signatures(A, "s0", s0_threshold="zero")
sB = core.join_irreducible_signatures(B, "s0", s0_threshold="zero")
def fmt(sigset):
    return sorted(tuple(sorted(int(t[0]) for t in s)) for s in sigset)
print("join_irred A:", fmt(sA))
print("join_irred B:", fmt(sB))
print("A - B:", fmt(sA - sB))
print("B - A:", fmt(sB - sA))
results["jirr_detail"] = {"A": fmt(sA), "B": fmt(sB),
                           "A_minus_B": fmt(sA - sB), "B_minus_A": fmt(sB - sA)}

results["experiment_complete"] = True
core.format_result("experiment_complete", True)
core.save_results(results, "/kaggle/working/phase1_p3diag_results.json")
print("Done.")
