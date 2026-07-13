# Concept Lattices as Canonical Feature Hierarchies

**Formal Concept Analysis as a Training-Free, Order-Theoretic Complement to Sparse Autoencoders for Mechanistic Interpretability**

A research framework that uses [Formal Concept Analysis (FCA)](https://en.wikipedia.org/wiki/Formal_concept_analysis) — a branch of order-theoretic mathematics built on Galois connections — to build deterministic, training-free concept lattices from neural network activations. The lattice's join-irreducible concepts serve as canonical features, providing a structural counterpart to Sparse Autoencoders (SAEs) that requires no gradient descent.

**Author:** Md Ariff Faysal Nayem

---

## Motivation

Sparse autoencoders have become the dominant tool for discovering interpretable features inside transformers, but they suffer from:

- **Seed-dependent irreproducibility** — different seeds produce different feature dictionaries
- **Hyperparameter sensitivity** — sparsity penalty, dictionary size, and architecture all change discovered features
- **Continuous-versus-discrete mismatch** — activations are continuous, but interpretable concepts are discrete
- **No canonical structure** — SAE features carry no algebraic relationships; any hierarchy must be discovered post-hoc

This framework offers a complementary approach: given a fixed scaling regime, the concept lattice is **uniquely determined** by the activation matrix alone — no random seeds, no training, no learned hyperparameters.

---

## Theoretical Contributions

| Contribution | Statement |
|---|---|
| **Proposition 1 (Canonicity)** | Given a scaling regime S ∈ {S0, S1, S2}, the concept lattice is uniquely determined up to isomorphism by the activation matrix. |
| **Proposition 2 (Order-invariance)** | Under strictly monotone activation transforms (e.g., x³, softplus), the concept lattice with quantile thresholds is bit-identical. |
| **Definition 1 (Structural polysemanticity)** | A neuron is structurally polysemantic if its attribute concepts are join-reducible in the concept lattice. |
| **Definition 2 (Superposition index)** | κ_S = 1 − σ_C(data) / σ_C(noise), measuring how far a model's discrete structure exceeds a calibrated noise baseline. |
| **Theorem (Galois embedding)** | Each SAE feature whose firing extent is not already an intersection of existing patterns corresponds to a join-irreducible of an extended concept lattice. |

---

## Key Results

All four pre-registered predictions hold on toy superposition models and MNIST:

| Claim | Setting | Result | Threshold | Status |
|---|---|---|---|---|
| **P1** κ_S separates trained models from noise | Toy (10 seeds x 3 sparsities) | κ_S = 0.83–0.97, **30/30 pass** | ≥ 0.10 | ✓ |
| **P2** Structural polysemanticity F1 vs ground truth | Toy (10 seeds) | F1 = 0.97 ± 0.05 | ≥ 0.80 | ✓ |
| **P3** Closed-concept Jaccard across data halves | Toy (5 splits) | **0.94** (vs SAE 0.80, raw join-irred 0.47) | ≥ 0.90 | ✓ |
| **P4** Order-invariance under monotone transforms | Toy | S1/S0 bit-identical under x³ and softplus | exact | ✓ |
| **Galois embedding** SAE features ↔ FCA concepts | Toy (10 seeds) | **82% ± 11%** of alive features embed | ≥ 50% | ✓ |
| **MNIST** Class-aligned concept discovery | MNIST MLP | FCA F1 0.52 vs k-means 0.16 | > baseline | ✓ |

On **Pythia-70M**, the framework as a feature-discovery method does not scale at this token budget — every token's firing pattern is unique. This is characterized as a precise negative result, establishing a quantitative scaling boundary via the discretization trilemma.

---

## Project Structure

```
├── src/
│   └── fca_core.py              # Core FCA library (NextClosure, Galois closures,
│                                 #   join-irreducibles, iceberg concepts, κ_S,
│                                 #   structural polysemanticity, superposition index)
├── templates/                   # Pristine notebook templates (checked in)
│   ├── phase1a.py               #   Smoke test
│   ├── phase1_toy.py            #   Toy superposition: P1 + P3 + P4
│   ├── phase1_mnist.py          #   MNIST MLP: FCA vs k-means
│   ├── phase1_p3diag.py         #   P3 reproducibility diagnostic
│   ├── phase2_sae.py            #   SAE Galois embedding + P2
│   ├── phase2_robust.py         #   Multi-seed robustness (P1, P2, theorem)
│   ├── phase2_p1rescue.py       #   κ_S regime sweep
│   ├── phase3_pythia.py         #   Pythia-70M residual stream
│   ├── phase3_pythia_v2.py      #   Pythia-70M MLP post-act
│   ├── phase3_track_b.py        #   Track B: FCA on SAE-firing context
│   └── phase3_track_b_v2.py     #   Track B: attribute implications + top-K lattice
├── notebooks/                   # Working copies (gitignored)
├── results/                     # Downloaded Kaggle outputs (gitignored)
├── paper/                       # LaTeX manuscript (Elsevier cas-dc template)
│   ├── main.tex                 #   Full paper
│   ├── main.pdf                 #   Compiled PDF
│   ├── refs.bib                 #   Bibliography (376 entries)
│   └── figs/                    #   5 paper figures + generation script
├── pipeline.py                  # CLI for Kaggle ML-Ops lifecycle
├── config.yaml                  # Pipeline configuration
├── proposal.md                  # Full research proposal
├── RESULTS.md                   # Experimental results scorecard
├── REPLICATION_GUIDE.md         # Step-by-step replication guide
└── requirements.txt             # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Kaggle account](https://www.kaggle.com/) with API access
- Kaggle API token (generate at [kaggle.com/settings](https://www.kaggle.com/settings))

### Installation

```bash
git clone https://github.com/mdariffaysalnayem/xai2.git
cd xai2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

Create a `.env` file at the repo root:

```
KAGGLE_API_TOKEN=KGAT_...
```

Edit `config.yaml` to set your Kaggle username and adjust pipeline settings as needed.

---

## Usage

The `pipeline.py` CLI manages the entire experiment lifecycle on Kaggle.

### Core workflow

```bash
# 1. Upload the core FCA library to Kaggle
python pipeline.py upload-src

# 2. Generate a working notebook from a template
python pipeline.py generate phase1_toy

# 3. (Optional) Edit notebooks/<phase>.py for parameter tweaks

# 4. Push the notebook to Kaggle as a kernel
python pipeline.py push notebooks/phase1_toy.py

# 5. Wait for completion (blocking, one line per state change)
python pipeline.py wait phase1_toy

# 6. Download results
python pipeline.py fetch phase1_toy

# 7. View results
python pipeline.py results phase1_toy
```

### All commands

| Command | Description |
|---|---|
| `upload-src` | Upload `src/*.py` to Kaggle as the core dataset |
| `push <notebook>` | Push a notebook to Kaggle as a kernel |
| `status [kernel\|phase]` | Print one-line kernel state |
| `wait <kernel\|phase>` | Blocking poll; returns 0 on success, non-zero on error |
| `fetch <kernel\|phase>` | Download kernel outputs to `results/` |
| `tail <kernel\|phase> [N]` | Last N lines of kernel log (default 40) |
| `results <phase>` | Pretty-print result JSON and grep `[RESULT]` lines |
| `generate <phase>` | Copy template to `notebooks/` |
| `upload-results <phase>` | Promote phase outputs to a Kaggle dataset |

### Iteration protocol

- **Notebook bug** → edit `notebooks/<phase>.py`, re-push
- **Core math/lib bug** → edit `src/fca_core.py`, run `upload-src`, wait ~5 min, re-push
- **Parameter issue** → tweak config in the notebook, re-push
- **After 5 retries** → stop and investigate

---

## Experimental Phases

### Phase 1 — Synthetic / Smoke Validation

| Notebook | Focus | Key Results |
|---|---|---|
| `phase1a` | Pipeline smoke test | ✓ |
| `phase1_toy` | Toy superposition (Elhage et al.) — P1, P3, P4 | P1 ✓, P3 ✓ (Jaccard 0.94), P4 ✓ (bit-identical) |
| `phase1_mnist` | MNIST MLP penultimate layer — FCA vs k-means | FCA F1 0.52 vs k-means 0.16 |
| `phase1_p3diag` | Join-irreducible fragility diagnostic | Raw Jaccard 0.47 vs closed-concept 0.94 |

### Phase 2 — SAE Comparison

| Notebook | Focus | Key Results |
|---|---|---|
| `phase2_sae` | Galois-embedding theorem + P2 (single seed) | P2 ✓ |
| `phase2_robust` | 10 seeds x 3 sparsities — headline claims | P1 ✓, P2 ✓, THM ✓ (82% embedding) |
| `phase2_p1rescue` | κ_S regime sweep — lock in S1 scaling | κ_S = 0.83–0.97 |

### Phase 3 — Pythia-70M (Characterized Negative Result)

| Notebook | Focus | Key Results |
|---|---|---|
| `phase3_pythia` | Residual stream layer 3 | 1 trivial concept; SAE density makes embedding metric unreliable |
| `phase3_pythia_v2` | MLP post-act + threshold sweep | 0 closed concepts at every threshold |
| `phase3_track_b` | Sparse SAE firings as attributes | 899 features, 3000 tokens — every token unique |
| `phase3_track_b_v2` | Attribute implications + top-32 lattice | 133 concepts (vs 1 Bernoulli) but Jaccard = 0.006 |

### Phase 4 — Vocabulary Aggregation

| Token Budget | Types | κ_S | Concepts | P3 Jaccard |
|---|---|---|---|---|
| 10k | 118 | 0.17 | 69 | 0.076 |
| 30k | 382 | 0.24 | 131 | 0.22 |
| 50k | 681 | 0.17 | 160 | 0.32 |
| 100k | 1,420 | 0.11 | 191 | 0.48 |

P_vocab2 and P_vocab3 pass; P_vocab1 falls short (0.48 < 0.90 threshold).

---

## Core Library: `fca_core.py`

The `src/fca_core.py` library (pure NumPy, no PyTorch dependency) provides:

**Scaling & Context Construction**
- `interordinal_context()` — builds binary formal context from continuous activations (S0/S1/S2)
- `clarify_context()` — removes duplicate attribute columns

**Galois Connection & Concept Enumeration**
- `closure_intent()` / `extent_of()` — Galois closure operators
- `enumerate_concepts()` — NextClosure algorithm (Ganter 1984) in lectic order

**Lattice Structure**
- `lower_covers()` — Hasse diagram lower covers
- `join_irreducibles()` / `meet_irreducibles()` — irreducible concept identification
- `irreducible_objects()` — Birkhoff fast object-reduction path

**Higher-Level Invariants**
- `frequent_closed_concepts()` — iceberg lattice (stability criterion)
- `superposition_index()` — σ_J and σ_C formulations
- `compression_ratio_kappa()` — κ_S against calibrated noise baseline
- `neuron_structural_polysemanticity()` — per-neuron structural polysemanticity detection
- `join_irreducible_signatures()` / `closed_concept_signatures()` — P3 comparators

**Result Emission**
- `format_result()` — greppable `[RESULT]` lines for automated success checking
- `save_results()` — JSON output to `/kaggle/working/`

---

## Scaling Regimes

| Regime | Description | Attributes | Use Case |
|---|---|---|---|
| **S0** | Binary (fire / no-fire at per-neuron quantile) | \|M\| = d | Fast, tractable baseline |
| **S1** | Ternary / 3-quantile interordinal | \|M\| ≈ 6d | Headline claims (canonicity, order-invariance) |
| **S2** | Full lossless interordinal + iceberg pruning | \|M\| ≈ 2d\|T\| | Exploratory; requires aggressive pruning |

All results report the regime used. S1 is the primary regime for published claims.

---

## Reproducing Results

See [`REPLICATION_GUIDE.md`](REPLICATION_GUIDE.md) for a detailed step-by-step guide. Quick summary:

1. Set up environment and Kaggle credentials (see Getting Started above)
2. `python pipeline.py upload-src`
3. For each phase, generate → push → wait → fetch → review results
4. All claims are backed by `[RESULT]` lines in kernel logs under `results/`

---

## Discretization Trilemma

Three desiderata pull against each other — all three cannot be simultaneously maximized:

```
        Losslessness
           /\
          /  \
         /    \
        / FCA  \
       /________\
Tractability  Canonicity
```

- **Losslessness** — exact recovery of activation values (gives exponential attribute count)
- **Tractability** — small, fixed attribute count (keeps lattice computable)
- **Canonicity** — thresholds determined by data alone (minimizes tuning)

The S0/S1/S2 ladder makes this tradeoff explicit and transparent rather than hiding it.

---

## Writing & Paper

The LaTeX manuscript is in `paper/main.tex` (Elsevier cas-dc double-column template). Key files:

- `paper/main.tex` — full manuscript
- `paper/refs.bib` — bibliography
- `paper/figs/make_figures.py` — generates all 5 paper figures
- `paper/figs/*.pdf` — pipeline overview, Galois embedding, reproducibility, Pythia, trilemma

---

## Citation

```bibtex
@article{nayem2025concept,
  title={Concept Lattices as Canonical Feature Hierarchies: A Training-Free, Order-Theoretic Complement to Sparse Autoencoders for Mechanistic Interpretability},
  author={Nayem, Md Ariff Faysal},
  year={2025}
}
```

---

## License

This project is a research codebase. Please contact the author for usage permissions.
