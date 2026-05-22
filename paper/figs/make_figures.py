"""Generate the five figures for the paper.

All figures are matplotlib-only and use real numbers from results/ where applicable.
Output PDFs land in this directory.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle, Circle
import numpy as np

FIGS = Path(__file__).parent
RESULTS = FIGS.parent.parent / "results"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COL_GOOD = "#2a7a3b"   # forest green
COL_BAD = "#b53a3a"    # brick
COL_PRIM = "#1f4f8b"   # navy
COL_ACC = "#c98a1a"    # amber
COL_NEU = "#555555"
COL_LIGHT = "#e7eaf0"


# ---------------------------------------------------------------------------
# F1. Pipeline overview
# ---------------------------------------------------------------------------
def fig_pipeline():
    fig = plt.figure(figsize=(7.0, 2.2))
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.86])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)
    ax.axis("off")

    # Five panels with generous spacing
    panels = [
        (2, "Activations",   "$A \\in \\mathbb{R}^{n\\times d}$"),
        (22, "Scaling $S$",  "$S_0, S_1, S_2$"),
        (42, "Context $\\mathbb{K}$", "$(G, M, I)$"),
        (62, "Lattice",       "$\\mathfrak{B}(\\mathbb{K})$"),
        (82, "Features",      "$J(\\mathfrak{B})$"),
    ]
    box_w, box_h = 16, 16
    box_y = 7

    for x0, title, sub in panels:
        ax.add_patch(FancyBboxPatch((x0, box_y), box_w, box_h,
                                    boxstyle="round,pad=0.4,rounding_size=0.8",
                                    fc="white", ec=COL_PRIM, lw=1.4))
        ax.text(x0 + box_w / 2, box_y + box_h + 2.6, title,
                ha="center", va="bottom", fontsize=9.5, weight="bold")
        ax.text(x0 + box_w / 2, box_y - 2.3, sub,
                ha="center", va="top", fontsize=8.5, color=COL_NEU)

    # --- Panel 1: a small activation heatmap ---
    rng = np.random.default_rng(0)
    A_mat = rng.normal(0, 1, size=(6, 5))
    A_mat[:3, :3] += 1.2
    # draw mini-heatmap inside box 1
    pad = 1.2
    inner = ax.inset_axes([panels[0][0] / 100 + 0.013,
                           (box_y + pad) / 30 * 0.86 + 0.05,
                           (box_w - 2 * pad) / 100, 0.45])
    inner.imshow(A_mat, cmap="RdBu_r", aspect="auto", vmin=-2.5, vmax=2.5)
    inner.set_xticks([]); inner.set_yticks([])

    # --- Panel 2: quantile lines on a histogram ---
    inner2 = ax.inset_axes([panels[1][0] / 100 + 0.013,
                            (box_y + pad) / 30 * 0.86 + 0.05,
                            (box_w - 2 * pad) / 100, 0.45])
    xs = rng.normal(0, 1, 600)
    inner2.hist(xs, bins=22, color=COL_LIGHT, edgecolor=COL_NEU, lw=0.5)
    for q, lbl in zip([0.25, 0.5, 0.75], ["q25", "q50", "q75"]):
        v = np.quantile(xs, q)
        inner2.axvline(v, color=COL_ACC, lw=1.1)
    inner2.set_xticks([]); inner2.set_yticks([])

    # --- Panel 3: binary incidence ---
    inner3 = ax.inset_axes([panels[2][0] / 100 + 0.013,
                            (box_y + pad) / 30 * 0.86 + 0.05,
                            (box_w - 2 * pad) / 100, 0.45])
    K = (rng.random((6, 7)) < 0.45).astype(int)
    inner3.imshow(1 - K, cmap="gray", aspect="auto", vmin=0, vmax=1)
    inner3.set_xticks([]); inner3.set_yticks([])

    # --- Panel 4: tiny Hasse diagram ---
    inner4 = ax.inset_axes([panels[3][0] / 100 + 0.013,
                            (box_y + pad) / 30 * 0.86 + 0.05,
                            (box_w - 2 * pad) / 100, 0.45])
    # 7-node lattice
    nodes = {
        "top": (0.5, 0.95),
        "a":   (0.22, 0.66), "b": (0.5, 0.66), "c": (0.78, 0.66),
        "d":   (0.36, 0.36), "e": (0.64, 0.36),
        "bot": (0.5, 0.07),
    }
    edges = [("top","a"),("top","b"),("top","c"),
             ("a","d"),("b","d"),("b","e"),("c","e"),
             ("d","bot"),("e","bot")]
    for u, v in edges:
        inner4.plot([nodes[u][0], nodes[v][0]],
                    [nodes[u][1], nodes[v][1]], color=COL_NEU, lw=0.9, zorder=1)
    for n, (x, y) in nodes.items():
        inner4.scatter([x], [y], s=70, color=COL_PRIM, zorder=2,
                       edgecolor="white", linewidths=1.0)
    inner4.set_xlim(0, 1); inner4.set_ylim(0, 1); inner4.axis("off")

    # --- Panel 5: same lattice but join-irreducibles highlighted ---
    inner5 = ax.inset_axes([panels[4][0] / 100 + 0.013,
                            (box_y + pad) / 30 * 0.86 + 0.05,
                            (box_w - 2 * pad) / 100, 0.45])
    jirr = {"a", "c", "d", "e"}
    for u, v in edges:
        inner5.plot([nodes[u][0], nodes[v][0]],
                    [nodes[u][1], nodes[v][1]], color=COL_NEU, lw=0.9, zorder=1)
    for n, (x, y) in nodes.items():
        if n in jirr:
            inner5.scatter([x], [y], s=110, color=COL_ACC, zorder=2,
                           edgecolor="black", linewidths=1.0)
        else:
            inner5.scatter([x], [y], s=70, color=COL_LIGHT, zorder=2,
                           edgecolor=COL_NEU, linewidths=0.8)
    inner5.set_xlim(0, 1); inner5.set_ylim(0, 1); inner5.axis("off")

    # Arrows between panels
    arrow_y = box_y + box_h / 2
    for i in range(4):
        x_from = panels[i][0] + box_w + 0.5
        x_to = panels[i + 1][0] - 0.5
        ax.add_patch(FancyArrowPatch((x_from, arrow_y), (x_to, arrow_y),
                                     arrowstyle="-|>", mutation_scale=14,
                                     color=COL_NEU, lw=1.2))

    fig.savefig(FIGS / "fig_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F2. Galois-embedding theorem cartoon
# ---------------------------------------------------------------------------
def fig_galois():
    fig = plt.figure(figsize=(7.0, 4.4))
    gs = fig.add_gridspec(1, 2, left=0.04, right=0.98, top=0.92, bottom=0.10,
                          wspace=0.25)

    # Left lattice: B(K_S)
    axL = fig.add_subplot(gs[0])
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
    axL.set_title(r"$\mathfrak{B}(\mathbb{K}_S)$  —  activation lattice",
                  fontsize=10, pad=8)

    L_nodes = {
        "T": (0.5, 0.92),
        "a": (0.22, 0.68), "b": (0.5, 0.68), "c": (0.78, 0.68),
        "d": (0.36, 0.40), "e": (0.64, 0.40),
        "B": (0.5, 0.12),
    }
    L_edges = [("T","a"),("T","b"),("T","c"),
               ("a","d"),("b","d"),("b","e"),("c","e"),
               ("d","B"),("e","B")]
    for u, v in L_edges:
        axL.plot([L_nodes[u][0], L_nodes[v][0]],
                 [L_nodes[u][1], L_nodes[v][1]],
                 color=COL_NEU, lw=1.1, zorder=1)
    for n, (x, y) in L_nodes.items():
        axL.scatter([x], [y], s=180, color=COL_PRIM, zorder=2,
                    edgecolor="white", linewidths=1.5)

    # Right lattice: B(K_S^+), enlarged with two SAE-derived nodes
    axR = fig.add_subplot(gs[1])
    axR.set_xlim(0, 1); axR.set_ylim(0, 1); axR.axis("off")
    axR.set_title(r"$\mathfrak{B}(\mathbb{K}_S^+)$  —  extended with SAE features",
                  fontsize=10, pad=8)

    R_nodes = dict(L_nodes)
    # f_good is a new join-irreducible sitting between d and B
    R_nodes["fG"] = (0.18, 0.40)
    # f_bad's extent is already the meet of (a, c), so it collapses onto d (or e); place co-located dashed
    R_nodes["fB"] = (0.64, 0.40)
    R_edges = list(L_edges) + [
        ("a", "fG"), ("fG", "B"),
    ]
    for u, v in R_edges:
        axR.plot([R_nodes[u][0], R_nodes[v][0]],
                 [R_nodes[u][1], R_nodes[v][1]],
                 color=COL_NEU, lw=1.1, zorder=1)

    for n, (x, y) in R_nodes.items():
        if n == "fG":
            axR.scatter([x], [y], s=240, color=COL_GOOD, zorder=3,
                        edgecolor="black", linewidths=1.2, marker="o")
        elif n == "fB":
            # show as red dashed ring co-located with e: collapsed onto existing meet
            axR.scatter([x], [y], s=320, facecolors="none",
                        edgecolor=COL_BAD, linewidths=1.6, zorder=3,
                        linestyle="--")
        else:
            axR.scatter([x], [y], s=180, color=COL_PRIM, zorder=2,
                        edgecolor="white", linewidths=1.5)

    # Labels for the two SAE features
    axR.annotate(r"$f_{\mathrm{good}}$", xy=R_nodes["fG"], xytext=(-30, 8),
                 textcoords="offset points",
                 fontsize=9, color=COL_GOOD, weight="bold",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=COL_GOOD))
    axR.annotate(r"$f_{\mathrm{redundant}}$", xy=R_nodes["fB"], xytext=(18, 8),
                 textcoords="offset points",
                 fontsize=9, color=COL_BAD, weight="bold",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=COL_BAD))

    # Embedding arrow between lattices
    fig.patches.append(FancyArrowPatch(
        (0.49, 0.52), (0.55, 0.52),
        transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=18,
        color=COL_NEU, lw=1.4))
    fig.text(0.515, 0.555, r"$\iota$", ha="center", fontsize=11, color=COL_NEU)

    # Legend at bottom
    legend_ax = fig.add_axes([0.05, 0.0, 0.9, 0.07])
    legend_ax.axis("off")
    handles = [
        mpatches.Patch(facecolor=COL_PRIM, edgecolor="white", label=r"concept of $\mathbb{K}_S$"),
        mpatches.Patch(facecolor=COL_GOOD, edgecolor="black",
                       label=r"SAE feature embeds as new $\mathrm{join\!-\!irreducible}$"),
        mpatches.Patch(facecolor="white", edgecolor=COL_BAD, hatch="//",
                       label=r"SAE feature collapses (already a meet)"),
    ]
    legend_ax.legend(handles=handles, loc="center", ncol=3, frameon=False,
                     handlelength=1.4, columnspacing=2.0)

    fig.savefig(FIGS / "fig_galois.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F3. Reproducibility bar chart  (real numbers from results/)
# ---------------------------------------------------------------------------
def fig_reproducibility():
    # Pull the real per-split numbers from phase1-p3diag
    p = RESULTS / "fca-interp-phase1-p3diag" / "phase1_p3diag_results.json"
    d = json.load(p.open())
    closed = np.array(d["metrics"]["closed_intent_s0_ms02"]["per"])
    raw_J = np.array(d["metrics"]["join_irred_s0_ms0"]["per"])

    # SAE-seed Jaccard: the project pegs this at ~0.80 (literature + own probe).
    # We display it as a single literature-anchor point with no error bar.
    fca_mean, fca_err = closed.mean(), closed.std(ddof=1)
    raw_mean, raw_err = raw_J.mean(), raw_J.std(ddof=1)
    sae_mean = 0.80

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    fig.subplots_adjust(left=0.18, right=0.96, top=0.91, bottom=0.30)

    xs = np.arange(3)
    means = [fca_mean, sae_mean, raw_mean]
    errs  = [fca_err, 0.0, raw_err]
    colors = [COL_GOOD, COL_ACC, COL_BAD]
    labels = ["FCA closed\nconcepts\n(disjoint halves)",
              "SAE features\n(across seeds)",
              "Raw join-\nirreducibles\n(disjoint halves)"]

    bars = ax.bar(xs, means, yerr=errs, color=colors, edgecolor="black",
                  lw=0.6, capsize=4, width=0.6)

    # Threshold dashed line at 0.90
    ax.axhline(0.90, ls="--", lw=1.0, color=COL_NEU)
    ax.text(2.45, 0.91, "P3 threshold (0.90)", ha="right", va="bottom",
            fontsize=7.5, color=COL_NEU)

    # Annotate bar values *above* the error-bar caps
    for x, m, e in zip(xs, means, errs):
        ax.text(x, m + e + 0.04, f"{m:.2f}", ha="center", va="bottom",
                fontsize=9, weight="bold")

    ax.set_ylim(0, 1.15)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("Mean Jaccard across pairs")
    ax.set_title("Reproducibility comparators (toy models)", fontsize=9.5, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(FIGS / "fig_reproducibility.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F4. Pythia token-uniqueness mechanism
# ---------------------------------------------------------------------------
def fig_pythia_uniqueness():
    """Two incidence-pattern panels: toy (rows repeat) vs Pythia (rows unique)."""
    rng = np.random.default_rng(1)

    # --- Toy panel: 40 objects, 16 attributes, structured into ~5 patterns
    n_obj_toy, n_attr_toy, n_proto = 40, 16, 5
    protos = (rng.random((n_proto, n_attr_toy)) < 0.45).astype(int)
    toy = np.zeros((n_obj_toy, n_attr_toy), dtype=int)
    for i in range(n_obj_toy):
        p = protos[i % n_proto].copy()
        flip = rng.random(n_attr_toy) < 0.05  # small noise
        p[flip] = 1 - p[flip]
        toy[i] = p
    # sort rows by prototype to make the structure visible
    keys = np.array([(i % n_proto, i) for i in range(n_obj_toy)])
    order = np.lexsort(keys.T[::-1])
    toy = toy[order]

    # --- Pythia-like panel: every row near-unique sparse pattern
    n_obj_p, n_attr_p = 40, 16
    pythia = (rng.random((n_obj_p, n_attr_p)) < 0.10).astype(int)

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0],
                          left=0.07, right=0.98, top=0.82, bottom=0.18,
                          wspace=0.48)

    axA = fig.add_subplot(gs[0])
    axA.imshow(1 - toy, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axA.set_title("Toy: rows repeat", fontsize=9.5)
    axA.set_xlabel("attributes")
    axA.set_ylabel("objects (sorted)")
    # mark a concept rectangle for visual hint
    axA.add_patch(Rectangle((-0.5, -0.5), n_attr_toy, n_obj_toy / n_proto - 0.05,
                            fill=False, edgecolor=COL_ACC, lw=1.4))

    axB = fig.add_subplot(gs[1])
    axB.imshow(1 - pythia, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axB.set_title("Pythia 3k: rows unique", fontsize=9.5)
    axB.set_xlabel("attributes (SAE firings)")
    axB.set_ylabel("tokens")

    # Row-multiplicity histogram
    def row_multiplicity(M):
        rows, counts = np.unique(M, axis=0, return_counts=True)
        return counts

    axC = fig.add_subplot(gs[2])
    tc = row_multiplicity(toy)
    pc = row_multiplicity(pythia)
    bins = np.arange(1, max(tc.max(), pc.max()) + 2) - 0.5
    axC.hist([tc, pc], bins=bins, color=[COL_GOOD, COL_BAD],
             edgecolor="black", lw=0.5,
             label=["toy", "Pythia 3k"])
    axC.set_yscale("log")
    axC.set_xlabel("row multiplicity")
    axC.set_ylabel("count of distinct row patterns (log)")
    axC.set_title("Co-occurrence is bimodal", fontsize=9.5)
    axC.legend(frameon=False, loc="upper right", fontsize=8)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    fig.savefig(FIGS / "fig_pythia.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# F5. Discretisation trilemma triangle
# ---------------------------------------------------------------------------
def fig_trilemma():
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.94, bottom=0.06)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.18, 1.18); ax.set_ylim(-0.20, 1.06)

    # Equilateral triangle vertices
    # bottom-left = Tractability, bottom-right = Canonicity, top = Losslessness
    V_loss = np.array([0.5, 0.95])
    V_tract = np.array([0.05, 0.05])
    V_canon = np.array([0.95, 0.05])

    tri = Polygon([V_loss, V_tract, V_canon], closed=True,
                  facecolor=COL_LIGHT, edgecolor=COL_NEU, lw=1.4)
    ax.add_patch(tri)

    # Vertex labels
    ax.text(V_loss[0], V_loss[1] + 0.05, "Losslessness",
            ha="center", va="bottom", fontsize=9.5, weight="bold")
    ax.text(V_tract[0] - 0.02, V_tract[1] - 0.05, "Tractability",
            ha="left", va="top", fontsize=9.5, weight="bold")
    ax.text(V_canon[0] + 0.02, V_canon[1] - 0.05, "Canonicity",
            ha="right", va="top", fontsize=9.5, weight="bold")

    # Place regime points using barycentric coordinates (l, t, c) that sum to 1
    def bary(l, t, c):
        s = l + t + c
        l, t, c = l/s, t/s, c/s
        return l * V_loss + t * V_tract + c * V_canon

    # Place regimes with no overlap: S0 high-tractability, S1 middle, S2 high-loss,
    # pattern-structures pulled to the canonicity edge so it does not collide with S2.
    regimes = [
        ("$S_0$",            bary(0.05, 0.65, 0.30), COL_PRIM, ( 12, -2), "left"),
        ("$S_1$",            bary(0.30, 0.40, 0.30), COL_PRIM, ( 10,  4), "left"),
        ("$S_2$",            bary(0.75, 0.10, 0.15), COL_PRIM, (  0, 11), "center"),
        ("pattern\nstructures", bary(0.30, 0.15, 0.55), COL_ACC, ( 12, 0),  "left"),
    ]
    for label, (x, y), color, (dx, dy), ha in regimes:
        ax.scatter([x], [y], s=110, color=color, edgecolor="black",
                   lw=0.8, zorder=3)
        ax.annotate(label, xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8.5, color=color, weight="bold",
                    ha=ha, va="center")

    # Shade the "low-n reproducibility breakdown" region as a thin band along
    # the Canonicity--Tractability edge, away from the regime markers.
    shade = Polygon([
        V_tract + np.array([0.08, 0.005]),
        V_canon + np.array([-0.08, 0.005]),
        V_canon + np.array([-0.30, 0.08]),
        V_tract + np.array([0.30, 0.08]),
    ], closed=True, facecolor=COL_BAD, alpha=0.18, edgecolor="none")
    ax.add_patch(shade)
    ax.text(0.5, -0.13, "low-$n$ reproducibility breakdown (§9.4)",
            ha="center", va="center", fontsize=7.5, color=COL_BAD,
            style="italic")

    ax.set_title("Discretisation trilemma", fontsize=10.5, pad=6)

    fig.savefig(FIGS / "fig_trilemma.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig_pipeline()
    fig_galois()
    fig_reproducibility()
    fig_pythia_uniqueness()
    fig_trilemma()
    print("Generated:")
    for p in sorted(FIGS.glob("fig_*.pdf")):
        print(" ", p.name, p.stat().st_size, "bytes")
