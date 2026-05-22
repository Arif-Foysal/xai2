# fca-interp Pipeline — Operating Guide

## Project
Formal Concept Analysis as a training-free, canonical counterpart to sparse autoencoders for mechanistic interpretability. The framework builds concept lattices from neural activations (Track A: interordinal scaling) and from SAE features (Track B), and tests pre-registered claims about polysemanticity, superposition, reproducibility, and order-invariance on toy models (Elhage et al. superposition), MNIST MLPs, GPT-2 small, and Pythia-70M. See `proposal.md` for the full mathematical scaffold, theorem/proposition statements (one theorem, two propositions, two definitions), the S0/S1/S2 scaling ladder, and the pre-registered predictions P1–P4.

## Architecture
- `src/fca_core.py` — core library, uploaded to Kaggle as dataset `mdariffaysalnayem/fca-interp-core`
- `pipeline.py` — CLI for push/status/fetch/upload-src/upload-results
- `config.yaml` — username, datasets, phases, success criteria
- `templates/` — pristine notebook templates (checked in)
- `notebooks/` — working copies you edit (gitignored)
- `results/` — downloaded Kaggle outputs (gitignored)

## Environment
The Kaggle API token lives in `.env` (gitignored) and is loaded automatically by `pipeline.py`. Just activate the venv:
```bash
source .venv/bin/activate
python pipeline.py push notebooks/<phase>.py    # token loads from .env
```
If `.env` is missing, create it with `KAGGLE_API_TOKEN=KGAT_...` from kaggle.com/settings.

## Running an experiment
1. `cp templates/<phase>.py notebooks/<phase>.py` (or `python pipeline.py generate <phase>`)
2. Edit `notebooks/<phase>.py` if needed (parameters, etc.)
3. `python pipeline.py push notebooks/<phase>.py`
4. `python pipeline.py wait <phase>` — single blocking call; prints one line per state change. Returns 0 on complete, non-zero on error/timeout.
5. `python pipeline.py fetch <phase>`
6. Read `results/<slug>/*.json` and grep `[RESULT]` lines to judge success. Use `python pipeline.py tail <phase>` for the last 40 log lines if you need to debug — never `cat` the full `.log`.

## Iteration protocol
When a notebook fails or success criteria aren't met:
- **Notebook bug** (syntax, shapes, logic) → edit `notebooks/<phase>.py`, re-push
- **Core math/lib bug** → edit `src/fca_core.py`, run `python pipeline.py upload-src`, **wait ~5 min**, re-push the notebook
- **Parameter issue** → tweak the notebook's config section, re-push
- **After 5 retries on the same notebook** → stop and ask the human

## Between phases
When phase N's outputs feed phase N+1:
1. `python pipeline.py upload-results phaseN`
2. Add the new dataset slug to `config.yaml` under `kaggle.datasets` (the CLI prints the line to add)
3. In `config.yaml` under the next notebook, add the key to its `datasets:` list

## Output conventions (do not break)
- `core.format_result(key, value)` — emits a greppable `[RESULT]` line
- `core.save_results(dict, path)` — writes JSON to `/kaggle/working/`
- Plots saved as PNG to `/kaggle/working/`

## Token-cost discipline (read this — it's why the harness exists)
- **Only three signals matter for "did this work?"**: `results/<slug>/*.json`, `[RESULT]` lines (use `pipeline.py results <phase>` or `pipeline.py fetch`), and the kernel state from `pipeline.py wait`. Don't Read raw `.log` files into context; use `pipeline.py tail <phase>` if you need recent log lines.
- **Use `wait`, not a polling loop.** One `python pipeline.py wait <phase>` is dramatically cheaper in context tokens than running `status` 6× in a row.
- **Keep `pipeline.py`, `src/fca_core.py`, `templates/` stable across iterations.** They are the cacheable prompt prefix; edit them only for durable changes. Iterate on `notebooks/<phase>.py`.
- **Don't print large arrays.** Truncate to a few items + length. `_pretty_print` already enforces this for fetched JSON.

## Project-specific guidance for FCA experiments
- **Lattice sizes blow up.** Always report raw and pruned `|L|` and `|J|` (join-irreducible count) as `[RESULT]` lines. If a scaling regime (S2 in particular) gives a lattice that doesn't fit in memory, drop to a higher iceberg support threshold and *say so* rather than silently changing the regime.
- **Scaling regime is a first-class result.** Every `[RESULT]` line tied to a lattice must include the regime (`s0` / `s1` / `s2`) and, for S2, the iceberg threshold `theta`. E.g. `[RESULT] num_join_irreducibles_s1 = 47`.
- **Pre-registered predictions are the gates.** P1 (σ ≥ 1.2 vs ≤ 1.05 Gaussian baseline), P2 (F1 ≥ 0.8 on toy / 50% on GPT-2), P3 (FCA-half Jaccard ≥ 0.90 vs SAE-seed Jaccard ≤ 0.80), P4 (FCA bit-identical under monotone transforms). See proposal §6.3. Each phase that targets a prediction should emit a `[RESULT] PX_met = true|false` line so advance/retry is a boolean grep.
- **Track A vs Track B.** Track A operates on discretized activations; Track B operates on binary SAE feature firings. Notebooks should declare which track in the header and prefix their result keys accordingly (`trackA_*`, `trackB_*`).

## When to stop and ask
- Phase boundary transitions (gate decisions)
- 5 retries exhausted
- Borderline / ambiguous results
- Anything that needs scientific judgement
- Before destructive actions (deleting kernels, force-pushing, etc.)

## Gotchas
- **Kaggle dataset propagation lag**: new dataset versions take ~5 minutes to mount in kernels. After `upload-src` or `upload-results`, wait before re-pushing.
- **`datasets create` silently no-ops** on existing datasets. The CLI handles this by trying `datasets version` first; don't bypass that pattern.
- **Mount paths drift**: always `os.walk` to find `fca_core.py` rather than hardcoding `/kaggle/input/<slug>/`.
- **GPU quota** is ~30h/week on free Kaggle. Set `gpu: false` in `config.yaml` for any notebook that doesn't need it. Phase 1 experiments (toy superposition, MNIST, small lattices) are all CPU-only.
- **Auth**: `pipeline.py` loads `KAGGLE_API_TOKEN` from `.env` automatically. If commands fail with 401/403, verify `.env` exists at repo root and contains a valid `KGAT_*` token.
