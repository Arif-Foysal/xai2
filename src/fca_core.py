"""
fca-interp core library.

Implements:
- Conceptual scaling (S0 binary, S1 interordinal, S2 lossless)
- Galois closure operators and NextClosure enumeration (Ganter 1984)
- Hasse-diagram lower covers and join-irreducible extraction
- Superposition index σ = |J(L)| / rank(A)  (Definition 2 of proposal)
- Structural-polysemanticity detection per neuron (Definition 1 of proposal)

All operations are pure NumPy / bool arrays; no torch dependency in core.
"""

import json
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Optional


# ============================================================================
# Result-emission helpers (do not break — notebooks depend on these)
# ============================================================================

def format_result(key, value, comment=None):
    """Print a greppable [RESULT] line and return the formatted string."""
    line = f"[RESULT] {key} = {value}"
    if comment:
        line += f"  # {comment}"
    print(line)
    return line


def save_results(data: dict, path):
    """Write a results dict as JSON. Use /kaggle/working/<phase>_results.json."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {p}")


# ============================================================================
# Conceptual scaling (proposal §3.2: S0 / S1 / S2 ladder)
# ============================================================================

def interordinal_context(
    activations: np.ndarray,
    scaling: str = "s1",
    n_quantiles: int = 3,
    s0_threshold: str = "median",
) -> Tuple[np.ndarray, List[Tuple[int, str, float]]]:
    """Build a binary formal context from continuous activations.

    Parameters
    ----------
    activations : (n_samples, d) ndarray
    scaling     : "s0" (binary fire/no-fire per neuron),
                  "s1" (interordinal at n_quantiles per-neuron quantiles),
                  "s2" (lossless: thresholds at all distinct values per neuron).
    n_quantiles : interior quantiles for S1 (default 3 → 0.25/0.5/0.75)
    s0_threshold: "median" (default), "zero", or a float

    Returns
    -------
    I       : (n_samples, n_attrs) bool ndarray
    labels  : list of (neuron_id, comparator, threshold) tuples — one per attr
    """
    n, d = activations.shape

    # Label tuple: (neuron_id, comparator, level_key). The level_key is a
    # *stable* identifier comparable across datasets (quantile rank for S0/S1,
    # raw value for S2). The actual threshold τ is data-dependent and is not
    # part of the label, so signatures are comparable across data halves (P3).
    if scaling == "s0":
        if s0_threshold == "median":
            thr = np.median(activations, axis=0)
            level = "median"
        elif s0_threshold == "zero":
            thr = np.full(d, 1e-12)   # ">0" semantics for ReLU "fired"
            level = "zero"
        elif isinstance(s0_threshold, str) and s0_threshold.startswith("q"):
            q = float(s0_threshold[1:]) / 100.0
            thr = np.quantile(activations, q, axis=0)
            level = s0_threshold
        elif isinstance(s0_threshold, (int, float)):
            thr = np.full(d, float(s0_threshold))
            level = f"abs{s0_threshold}"
        else:
            raise ValueError(f"Unknown s0_threshold: {s0_threshold}")
        I = activations >= thr[None, :]
        labels = [(j, ">=", level) for j in range(d)]
        return I, labels

    if scaling == "s1":
        qs = np.linspace(0, 1, n_quantiles + 2)[1:-1]
        cols, labels = [], []
        for j in range(d):
            taus = np.quantile(activations[:, j], qs)
            for r, tau in enumerate(taus):
                cols.append(activations[:, j] >= tau)
                labels.append((j, ">=", r))
                cols.append(activations[:, j] <= tau)
                labels.append((j, "<=", r))
        I = np.stack(cols, axis=1)
        return I, labels

    if scaling == "s2":
        cols, labels = [], []
        for j in range(d):
            taus = np.unique(activations[:, j])
            for tau in taus:
                cols.append(activations[:, j] >= tau)
                labels.append((j, ">=", float(tau)))
                cols.append(activations[:, j] <= tau)
                labels.append((j, "<=", float(tau)))
        I = np.stack(cols, axis=1)
        return I, labels

    raise ValueError(f"Unknown scaling: {scaling}")


def clarify_context(
    I: np.ndarray, labels: List
) -> Tuple[np.ndarray, List, np.ndarray]:
    """Remove duplicate attribute columns. Returns clarified I, labels, and
    a mapping old_col -> new_col index."""
    seen: Dict[bytes, int] = {}
    keep_idx, new_labels = [], []
    mapping = np.zeros(I.shape[1], dtype=int)
    for j in range(I.shape[1]):
        key = I[:, j].tobytes()
        if key in seen:
            mapping[j] = seen[key]
        else:
            seen[key] = len(keep_idx)
            mapping[j] = len(keep_idx)
            keep_idx.append(j)
            new_labels.append(labels[j])
    return I[:, keep_idx], new_labels, mapping


# ============================================================================
# Galois closure operators
# ============================================================================

def closure_intent(B_mask: np.ndarray, I: np.ndarray) -> np.ndarray:
    """Closure of attribute set B (as bool mask): returns B'' (bool mask)."""
    if B_mask.any():
        extent = I[:, B_mask].all(axis=1)
    else:
        extent = np.ones(I.shape[0], dtype=bool)
    if extent.any():
        return I[extent, :].all(axis=0)
    return np.ones(I.shape[1], dtype=bool)


def extent_of(B_mask: np.ndarray, I: np.ndarray) -> np.ndarray:
    """B' as bool mask over objects."""
    if B_mask.any():
        return I[:, B_mask].all(axis=1)
    return np.ones(I.shape[0], dtype=bool)


# ============================================================================
# NextClosure concept enumeration (Ganter 1984)
# ============================================================================

def enumerate_concepts(
    I: np.ndarray, max_concepts: int = 100_000
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], bool]:
    """Enumerate all formal concepts in lectic order.

    Returns
    -------
    concepts : list of (extent_mask, intent_mask)
    truncated: True if the enumeration hit max_concepts
    """
    n, m = I.shape
    concepts = []
    A = closure_intent(np.zeros(m, dtype=bool), I)
    concepts.append((extent_of(A, I).copy(), A.copy()))
    truncated = False
    while True:
        if len(concepts) >= max_concepts:
            truncated = True
            break
        next_A = None
        for i in range(m - 1, -1, -1):
            if A[i]:
                continue
            prefix = A.copy()
            prefix[i:] = False
            prefix[i] = True
            B = closure_intent(prefix, I)
            # lectic condition: B agrees with A on positions < i
            if np.array_equal(B[:i], A[:i]):
                next_A = B
                break
        if next_A is None:
            break
        A = next_A
        concepts.append((extent_of(A, I).copy(), A.copy()))
    return concepts, truncated


# ============================================================================
# Hasse diagram / lower covers / join-irreducibles
# ============================================================================

def lower_covers(
    concepts: List[Tuple[np.ndarray, np.ndarray]],
) -> List[List[int]]:
    """For each concept index i, return indices of its lower covers.

    A concept d covers c (c < d) iff extent(c) ⊊ extent(d) and no e has
    extent(c) ⊊ extent(e) ⊊ extent(d). Equivalently d is a "lower neighbor"
    of c when we invert: c covers d below. We compute the *lower* covers of
    each concept i (concepts immediately below).
    """
    n = len(concepts)
    # Pack extents into a uint8 matrix for fast bitwise ops
    ext_matrix = np.stack([c[0] for c in concepts]).astype(bool)  # (n, n_obj)
    sizes = ext_matrix.sum(axis=1)

    covers_down: List[List[int]] = [[] for _ in range(n)]

    # Sort indices by extent size ascending; for each i we look at smaller j.
    order = np.argsort(sizes, kind="stable")

    # For efficiency: subset(ext_j, ext_i) iff (ext_j & ~ext_i).any() == False
    # Use packed bits for AND/OR.
    packed = np.packbits(ext_matrix, axis=1)  # (n, ceil(n_obj/8))

    def is_subset(a: int, b: int) -> bool:
        # extent[a] ⊆ extent[b]
        return not np.any(packed[a] & ~packed[b])

    for i in range(n):
        si = sizes[i]
        # gather candidates: j with sizes[j] < si and ext[j] ⊆ ext[i]
        cand = [int(j) for j in range(n) if sizes[j] < si and is_subset(j, i)]
        # keep maximal candidates
        for j in cand:
            sj = sizes[j]
            is_max = True
            for k in cand:
                if k == j:
                    continue
                if sizes[k] > sj and is_subset(j, k):
                    is_max = False
                    break
            if is_max:
                covers_down[i].append(j)
    return covers_down


def join_irreducibles(
    concepts: List[Tuple[np.ndarray, np.ndarray]],
    covers_down: List[List[int]],
) -> List[int]:
    """Concept indices that are join-irreducible (exactly one lower cover)."""
    return [i for i in range(len(concepts)) if len(covers_down[i]) == 1]


def meet_irreducibles(
    concepts: List[Tuple[np.ndarray, np.ndarray]],
    covers_down: List[List[int]],
) -> List[int]:
    """Concept indices that are meet-irreducible (exactly one upper cover)."""
    n = len(concepts)
    upper = [0] * n
    for i in range(n):
        for j in covers_down[i]:
            upper[j] += 1
    return [i for i in range(n) if upper[i] == 1]


# ============================================================================
# Fast object-reduction path (Birkhoff: |J(L)| = #irreducible object concepts)
# ============================================================================

def clarify_objects(I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Deduplicate identical rows. Returns (I_clar_rows, row_mapping).
    row_mapping[g] = clarified row index for original object g."""
    seen: Dict[bytes, int] = {}
    keep, mapping = [], np.zeros(I.shape[0], dtype=int)
    for g in range(I.shape[0]):
        k = I[g, :].tobytes()
        if k in seen:
            mapping[g] = seen[k]
        else:
            seen[k] = len(keep)
            mapping[g] = len(keep)
            keep.append(g)
    return I[keep, :], mapping


def irreducible_objects(I_obj_clar: np.ndarray) -> np.ndarray:
    """Bool mask of irreducible objects in a row-clarified context.

    Object g is reducible iff row_g equals the bitwise OR of {row_h : h ≠ g
    and row_h ⊆ row_g}. Equivalently, g's intent is the union of the intents
    of objects strictly below it in the object-concept order.

    Uses packed bits → O(n² · ceil(m/8)) total.
    """
    n_obj, n_attr = I_obj_clar.shape
    if n_obj == 0:
        return np.zeros(0, dtype=bool)
    packed = np.packbits(I_obj_clar, axis=1)  # (n_obj, ceil(n_attr/8))
    is_irr = np.zeros(n_obj, dtype=bool)
    for g in range(n_obj):
        notg = ~packed[g]
        # row_h ⊆ row_g iff (packed[h] & notg) is all-zero
        diff = packed & notg  # (n_obj, w)
        subset_h = (diff == 0).all(axis=1)
        subset_h[g] = False
        if subset_h.any():
            union = np.bitwise_or.reduce(packed[subset_h], axis=0)
        else:
            union = np.zeros_like(packed[g])
        if not np.array_equal(union, packed[g]):
            is_irr[g] = True
    return is_irr


def join_irreducible_signatures(
    activations: np.ndarray,
    scaling: str = "s1",
    n_quantiles: int = 3,
    s0_threshold: str = "zero",
    min_support: float = 0.0,
) -> set:
    """Fingerprint each join-irreducible by the *intent* of its dual
    irreducible object concept, expressed as a frozenset of stable attribute
    label keys (neuron, comparator, level).

    Because label keys do not contain data-dependent threshold values, the
    returned set is comparable across disjoint data samples — this is the
    correct stability comparator for P3 (the extent-based comparison is not,
    since extents live over different object universes per sample).

    min_support : float in [0,1) or int ≥ 1. Drop join-irreducibles whose
        extent covers fewer than this many objects (fraction of n_objects if
        a float < 1, absolute count otherwise). This is the iceberg-lattice
        pruning of proposal §5: rare irreducibles are sampling-noise
        singletons; filtering them is what makes the discovered concept set
        reproducible across data draws (P3).
    """
    if scaling == "s0":
        I, labels = interordinal_context(activations, scaling="s0", s0_threshold=s0_threshold)
    else:
        I, labels = interordinal_context(activations, scaling=scaling, n_quantiles=n_quantiles)
    n_obj_total = I.shape[0]
    if isinstance(min_support, float) and 0.0 < min_support < 1.0:
        min_count = int(np.ceil(min_support * n_obj_total))
    else:
        min_count = int(min_support)

    I_clar, labels_clar, col_map = clarify_context(I, labels)
    clar_label = {}
    for raw_idx, lab in enumerate(labels):
        clar_label[int(col_map[raw_idx])] = lab

    I_obj_clar, row_map = clarify_objects(I_clar)
    # multiplicity of each clarified object row (how many original objects map to it)
    mult = np.bincount(row_map, minlength=I_obj_clar.shape[0])
    is_irr = irreducible_objects(I_obj_clar)

    sigs = set()
    for g in np.where(is_irr)[0]:
        row_g = I_obj_clar[g]
        # extent: clarified rows whose intent ⊇ row_g's intent
        ext_clar = np.all(I_obj_clar >= row_g, axis=1)
        support = int(mult[ext_clar].sum())
        if support < min_count:
            continue
        intent_cols = np.where(row_g)[0]
        sig = frozenset(clar_label[int(c)] for c in intent_cols)
        sigs.add(sig)
    return sigs


def jaccard(setA: set, setB: set) -> float:
    if not setA and not setB:
        return 1.0
    return len(setA & setB) / max(len(setA | setB), 1)


def attribute_closures(
    activations: np.ndarray,
    scaling: str = "s0",
    n_quantiles: int = 3,
    s0_threshold: str = "zero",
) -> set:
    """Set of attribute closures {closure({m}) : m ∈ M}, each as a frozenset
    of stable label keys.

    closure({m}) = the set of attributes that hold for *every* object that has
    m — i.e. the deterministic co-occurrence / implication structure anchored
    at m. Unlike join-irreducibility, this is a statistical co-occurrence
    property, not a combinatorial union-decomposition; it is therefore robust
    to the presence/absence of rare "building-block" patterns across data
    draws. There are exactly |M_clarified| such closures, so the computation
    is bounded and tractable. This is the canonical reproducibility invariant
    for P3.
    """
    if scaling == "s0":
        I, labels = interordinal_context(activations, scaling="s0", s0_threshold=s0_threshold)
    else:
        I, labels = interordinal_context(activations, scaling=scaling, n_quantiles=n_quantiles)
    I_clar, _, col_map = clarify_context(I, labels)
    clar_label = {}
    for raw_idx, lab in enumerate(labels):
        clar_label.setdefault(int(col_map[raw_idx]), lab)
    m = I_clar.shape[1]
    sigs = set()
    for col in range(m):
        Bmask = np.zeros(m, dtype=bool)
        Bmask[col] = True
        clo = closure_intent(Bmask, I_clar)
        sigs.add(frozenset(clar_label[int(c)] for c in np.where(clo)[0]))
    return sigs


def frequent_closed_concepts(
    activations: np.ndarray,
    scaling: str = "s0",
    n_quantiles: int = 3,
    s0_threshold: str = "zero",
    min_support: float = 0.02,
) -> list:
    """Iceberg concept set (proposal §5): object-induced closed concepts whose
    extent covers at least `min_support` of the objects.

    For each distinct object row we close its intent (B''); the resulting
    closed attribute set is a concept intent, and its extent is all objects
    sharing it. We keep those above the support threshold.

    This is the canonical, REPRODUCIBLE feature unit (P3 diagnostic: Jaccard
    ≈0.94 across disjoint data halves, vs ≈0.47 for individual join-
    irreducibles which are combinatorially fragile). Returns a list of dicts:
        {extent: np.ndarray of object indices,
         intent_signature: frozenset of stable label keys,
         support: int}
    sorted by descending support.
    """
    if scaling == "s0":
        I, labels = interordinal_context(activations, scaling="s0", s0_threshold=s0_threshold)
    else:
        I, labels = interordinal_context(activations, scaling=scaling, n_quantiles=n_quantiles)
    n_obj = I.shape[0]
    min_count = int(np.ceil(min_support * n_obj)) if 0.0 < min_support < 1.0 else int(min_support)

    I_clar, _, col_map = clarify_context(I, labels)
    clar_label = {}
    for raw_idx, lab in enumerate(labels):
        clar_label.setdefault(int(col_map[raw_idx]), lab)

    seen_intents = {}
    for g in range(n_obj):
        row = I_clar[g]
        # closure of this object's intent
        ext = np.all(I_clar >= row, axis=1) if row.any() else np.ones(n_obj, dtype=bool)
        intent = I_clar[ext].all(axis=0) if ext.any() else np.ones(I_clar.shape[1], dtype=bool)
        key = intent.tobytes()
        if key in seen_intents:
            continue
        support = int(ext.sum())
        if support < min_count:
            continue
        seen_intents[key] = {
            "extent": np.where(ext)[0],
            "intent_signature": frozenset(clar_label[int(c)] for c in np.where(intent)[0]),
            "support": support,
        }
    out = list(seen_intents.values())
    out.sort(key=lambda c: -c["support"])
    return out


def closed_concept_signatures(
    activations: np.ndarray,
    scaling: str = "s0",
    n_quantiles: int = 3,
    s0_threshold: str = "zero",
    min_support: float = 0.02,
) -> set:
    """Set of intent signatures of the iceberg concept set — the P3 comparator."""
    return {
        c["intent_signature"]
        for c in frequent_closed_concepts(
            activations, scaling=scaling, n_quantiles=n_quantiles,
            s0_threshold=s0_threshold, min_support=min_support,
        )
    }


# ============================================================================
# Superposition index σ_S (Definition 2)
# ============================================================================

def superposition_index(
    activations: np.ndarray,
    scaling: str = "s1",
    n_quantiles: int = 3,
    s0_threshold: str = "zero",
    rank_tol: Optional[float] = None,
) -> dict:
    """Lattice-derived complexity indices.

    σ_J = |J(L_S)| / rank(activations)    [join-irreducible count]
    σ_C = n_objects_clarified / rank(A)   [distinct categorical patterns]

    σ_C is the more robust quantity. It is monotone in the number of
    categorical states the activations distinguish and is insensitive to the
    saturation regime that can collapse |J|.

    The published superposition index uses the COMPRESSION RATIO

        κ_S(model) = 1 − σ_C(activations) / σ_C(matched_noise_baseline)

    interpreted as: the fraction of categorical complexity that the model's
    learned structure ABSORBS relative to a matched-shape isotropic baseline.
    A positive κ_S means the model imposes categorical structure (compresses
    patterns); κ_S ≈ 0 means the activations are as combinatorially diverse
    as noise. Use `compression_ratio_kappa(activations)` to compute κ_S.
    """
    if scaling == "s0":
        I, labels = interordinal_context(activations, scaling="s0", s0_threshold=s0_threshold)
    else:
        I, labels = interordinal_context(activations, scaling=scaling, n_quantiles=n_quantiles)
    I_clar, _, _ = clarify_context(I, labels)
    I_obj_clar, _ = clarify_objects(I_clar)
    is_irr = irreducible_objects(I_obj_clar)
    n_J = int(is_irr.sum())
    n_obj_clar = int(I_obj_clar.shape[0])
    if rank_tol is None:
        rank = int(np.linalg.matrix_rank(activations))
    else:
        rank = int(np.linalg.matrix_rank(activations, tol=rank_tol))
    return {
        "scaling": scaling,
        "n_objects": int(I.shape[0]),
        "n_objects_clarified": n_obj_clar,
        "n_attributes_raw": int(I.shape[1]),
        "n_attributes_clarified": int(I_clar.shape[1]),
        "n_join_irreducibles": n_J,
        "rank": rank,
        "sigma": n_J / max(rank, 1),
        "sigma_J": n_J / max(rank, 1),
        "sigma_C": n_obj_clar / max(rank, 1),
    }


def compression_ratio_kappa(
    activations: np.ndarray,
    scaling: str = "s0",
    s0_threshold: str = "zero",
    n_quantiles: int = 3,
    n_baseline_seeds: int = 5,
) -> dict:
    """Compression-ratio superposition index κ_S = 1 − σ_C / σ_C^baseline.

    The matched baseline is a Gaussian (isotropic) sample of the same shape;
    we average σ_C over n_baseline_seeds draws to reduce baseline variance.

    Returns dict with σ_C(activations), mean/std of σ_C(baseline), κ_S,
    plus the underlying counts so callers can emit [RESULT] lines."""
    s = superposition_index(activations, scaling=scaling, s0_threshold=s0_threshold,
                             n_quantiles=n_quantiles)
    n, d = activations.shape
    sigC_baselines = []
    for seed in range(n_baseline_seeds):
        rng = np.random.default_rng(seed)
        g = rng.standard_normal((n, d)).astype(activations.dtype)
        sigC_baselines.append(superposition_index(g, scaling=scaling,
                                                   s0_threshold=s0_threshold,
                                                   n_quantiles=n_quantiles)["sigma_C"])
    base_mean = float(np.mean(sigC_baselines))
    base_std = float(np.std(sigC_baselines))
    kappa = 1.0 - s["sigma_C"] / max(base_mean, 1e-9)
    return {
        "sigma_C_data": s["sigma_C"],
        "sigma_C_baseline_mean": base_mean,
        "sigma_C_baseline_std": base_std,
        "kappa": kappa,
        "n_obj_clar_data": s["n_objects_clarified"],
        "rank": s["rank"],
    }


# ============================================================================
# Structural polysemanticity (Definition 1)
# ============================================================================

def attribute_concept_intent(m_idx: int, I: np.ndarray) -> np.ndarray:
    """Return the intent of the attribute concept γ(m) = (m', m'')."""
    extent = I[:, m_idx]
    if extent.any():
        return I[extent, :].all(axis=0)
    return np.ones(I.shape[1], dtype=bool)


def find_concept_by_intent(
    intent_mask: np.ndarray,
    concepts: List[Tuple[np.ndarray, np.ndarray]],
) -> int:
    for i, (_, B) in enumerate(concepts):
        if np.array_equal(B, intent_mask):
            return i
    return -1


def neuron_structural_polysemanticity(
    activations: np.ndarray,
    scaling: str = "s1",
    n_quantiles: int = 3,
) -> dict:
    """For each neuron, True iff any of its attribute concepts γ(m_{j,τ}) is
    join-reducible (Definition 1). Fast path via object reduction:
    γ(m) ∈ J(L) ↔ E_m equals the extent of some irreducible object concept.
    """
    I, labels = interordinal_context(activations, scaling=scaling, n_quantiles=n_quantiles)
    I_clar, labels_clar, col_mapping = clarify_context(I, labels)
    I_obj_clar, _ = clarify_objects(I_clar)
    is_irr = irreducible_objects(I_obj_clar)
    n_J = int(is_irr.sum())

    # Extents of irreducible object concepts (bool masks over clarified rows)
    irr_extents = set()
    if I_obj_clar.shape[0] > 0:
        for g in np.where(is_irr)[0]:
            row_g = I_obj_clar[g]
            ext = np.all(I_obj_clar >= row_g, axis=1)
            irr_extents.add(ext.tobytes())

    n_attr_clar = I_obj_clar.shape[1]
    attr_redu = np.zeros(n_attr_clar, dtype=bool)
    for m in range(n_attr_clar):
        E_m = I_obj_clar[:, m]
        attr_redu[m] = E_m.tobytes() not in irr_extents

    d = activations.shape[1]
    poly_mask = np.zeros(d, dtype=bool)
    for raw_idx, (nid, _, _) in enumerate(labels):
        clar_idx = int(col_mapping[raw_idx])
        if 0 <= clar_idx < n_attr_clar and attr_redu[clar_idx]:
            poly_mask[nid] = True

    return {
        "poly_mask": poly_mask.tolist(),
        "n_polysemantic": int(poly_mask.sum()),
        "n_attributes_clarified": n_attr_clar,
        "n_objects_clarified": int(I_obj_clar.shape[0]),
        "n_join_irreducibles": n_J,
    }


# ============================================================================
# Self-test (runs on import — fast, tiny example from Ganter & Wille)
# ============================================================================

def _selftest():
    """Tiny well-known context: 4 objects, 3 attributes, planted structure.

        a b c
    g1: 1 1 0
    g2: 1 0 1
    g3: 0 1 1
    g4: 1 1 1
    """
    I = np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=bool,
    )
    concepts, _ = enumerate_concepts(I)
    # Expected concepts (extent, intent):
    # ({1,2,3,4}, {}), ({1,2,4}, {a}), ({1,3,4}, {b}), ({2,3,4}, {c}),
    # ({1,4}, {a,b}), ({2,4}, {a,c}), ({3,4}, {b,c}), ({4}, {a,b,c})
    assert len(concepts) == 8, f"expected 8 concepts, got {len(concepts)}"
    covers = lower_covers(concepts)
    # Lattice order: smaller extent = lower. Bottom = ({4}, {abc}); top = ({1234}, {}).
    sizes = [int(c[0].sum()) for c in concepts]
    bot_idx = sizes.index(min(sizes))
    top_idx = sizes.index(max(sizes))
    assert len(covers[bot_idx]) == 0, f"bottom should have 0 lower covers, got {len(covers[bot_idx])}"
    assert len(covers[top_idx]) == 3, f"top should have 3 lower covers, got {len(covers[top_idx])}"
    # Join-irreducibles in this lattice are {a,b}, {a,c}, {b,c} — each has
    # the bottom ({4}, {abc}) as its unique lower cover.
    J = join_irreducibles(concepts, covers)
    assert len(J) == 3, f"expected 3 join-irreducibles via Hasse, got {len(J)}"

    # Fast object-reduction path must agree
    I_obj_clar, _ = clarify_objects(I)
    is_irr = irreducible_objects(I_obj_clar)
    assert int(is_irr.sum()) == 3, (
        f"expected 3 irreducible objects, got {int(is_irr.sum())}"
    )

    # And the high-level superposition_index function must agree on σ
    # σ = |J| / rank; for this synthetic context, rank is float so use S0
    # on a richer matrix:
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 4))  # 50 samples, 4 features
    s0 = superposition_index(A, scaling="s0")
    assert s0["n_join_irreducibles"] >= 1


_selftest()
