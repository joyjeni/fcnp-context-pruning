"""Generate publication-quality figures for the FCNP paper.

Outputs (PNG @ 300 dpi + SVG) into figures/:
  fig1_architecture.{png,svg}      System pipeline: Kaggle -> metrics.json -> Vercel
  fig2_algorithm_flow.{png,svg}    FCNP iteration loop (Kirchhoff)
  fig3_novelty.{png,svg}           FCNP vs SOTA families (axes diagram)
  fig4_context_graph.{png,svg}     Before / after pruning on the context graph
"""
from __future__ import annotations

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

OUT = os.path.dirname(os.path.abspath(__file__))

# Consistent paper palette (color-blind safe, IEEE-print friendly)
C_PRIMARY = "#1F3A5F"   # deep navy
C_ACCENT = "#C44536"    # brick red
C_OK = "#2E7D32"        # green
C_MUTED = "#6C757D"     # slate
C_LIGHT = "#E8ECF1"     # near-white panel
C_EDGE = "#283142"
C_HIGHLIGHT = "#F5B400" # amber

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(os.path.join(OUT, f"{name}.svg"), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 1 : System architecture (Kaggle -> metrics -> Vercel dashboard)
# ---------------------------------------------------------------------------
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    ax.set_title("FCNP Reproducibility Pipeline: Kaggle Benchmark $\\rightarrow$ Metrics Artifact $\\rightarrow$ Vercel Dashboard",
                 loc="left", pad=12)

    def box(x, y, w, h, label, sub=None, color=C_LIGHT, edge=C_EDGE, bold=True):
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                               linewidth=1.4, edgecolor=edge, facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2 + (0.18 if sub else 0), label,
                ha="center", va="center", fontsize=10,
                fontweight="bold" if bold else "normal", color=C_EDGE)
        if sub:
            ax.text(x + w/2, y + h/2 - 0.22, sub, ha="center", va="center",
                    fontsize=8.5, color=C_MUTED, style="italic")

    def arrow(x1, y1, x2, y2, label=None, color=C_PRIMARY):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                            linewidth=1.6, color=color)
        ax.add_patch(a)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.16, label, ha="center",
                    fontsize=8.5, color=color, fontweight="bold")

    # Stage 1: Kaggle environment
    box(0.2, 2.6, 2.6, 1.7, "Kaggle Notebook",
        sub="all-MiniLM-L6-v2 (384-d)\nGPU runtime", color="#EAF1FB")
    box(0.4, 0.6, 2.2, 1.5, "ToolBench Dataset",
        sub="16k queries\n(Qin et al., 2024)", color="#FFF7E0")

    arrow(1.5, 2.1, 1.5, 2.6)

    # Stage 2: FCNP core
    box(3.4, 2.6, 2.6, 1.7, "FCNP Core",
        sub="Kirchhoff iteration\n$D(t{+}1)=(1{-}\\mu)D{+}\\alpha|Q|^{\\gamma}$",
        color="#F0E5F2")
    box(3.4, 0.6, 2.6, 1.5, "Baselines",
        sub="BM25 / Selective-Ctx\nLLMLingua / RECOMP", color=C_LIGHT)

    arrow(3.0, 3.45, 3.4, 3.45, "context")
    arrow(2.7, 1.35, 3.4, 1.35)

    # Stage 3: Evaluation
    box(6.6, 2.6, 2.4, 1.7, "Evaluation Harness",
        sub="Recall / F1 / nDCG\nbootstrap-CI, Wilcoxon",
        color="#EAF1FB")
    arrow(6.0, 3.45, 6.6, 3.45)
    arrow(6.0, 1.35, 6.6, 2.8)

    # Stage 4: Artifact
    box(6.6, 0.6, 2.4, 1.4, "metrics.json",
        sub="results.csv / summary.md", color="#FFF7E0")
    arrow(7.8, 2.6, 7.8, 2.0)

    # Stage 5: Vercel dashboard
    box(9.4, 0.6, 1.5, 3.7, "Vercel\nDashboard",
        sub="Next.js 14\nRecharts", color="#E6F4EA")
    # Horizontal POST arrow with label cleanly above
    arrow(9.0, 1.3, 9.4, 1.3, color=C_ACCENT)
    ax.text(9.2, 1.55, "POST /api/metrics", ha="center", va="bottom",
            fontsize=8.0, color=C_ACCENT, fontweight="bold")
    arrow(9.0, 3.45, 9.4, 3.45, color=C_OK)

    # Legend
    leg = [
        mpatches.Patch(facecolor="#EAF1FB", edgecolor=C_EDGE, label="Compute"),
        mpatches.Patch(facecolor="#F0E5F2", edgecolor=C_EDGE, label="Method"),
        mpatches.Patch(facecolor="#FFF7E0", edgecolor=C_EDGE, label="Data"),
        mpatches.Patch(facecolor="#E6F4EA", edgecolor=C_EDGE, label="Serving"),
    ]
    ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.0, -0.05),
              ncol=4, frameon=False, fontsize=9)

    save(fig, "fig1_architecture")


# ---------------------------------------------------------------------------
# Figure 2 : FCNP algorithm flow (Kirchhoff iteration loop)
# ---------------------------------------------------------------------------
def fig_algorithm_flow():
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    ax.set_title("FCNP Algorithm: Kirchhoff Iteration Loop",
                 loc="left", pad=10)

    def stage(x, y, w, h, top, formula=None, color="#EAF1FB"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    linewidth=1.4, edgecolor=C_EDGE, facecolor=color))
        ax.text(x+w/2, y+h-0.28, top, ha="center", va="top",
                fontsize=10, fontweight="bold", color=C_EDGE)
        if formula:
            ax.text(x+w/2, y+h/2-0.15, formula, ha="center", va="center",
                    fontsize=9.5, color=C_PRIMARY)

    def arrow(x1, y1, x2, y2, color=C_PRIMARY, label=None, curved=False, rad=0.0):
        style = "arc3,rad=%.2f" % rad if curved else "arc3,rad=0"
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=14, linewidth=1.6,
                            color=color, connectionstyle=style)
        ax.add_patch(a)
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.18, label, ha="center",
                    fontsize=8.5, color=color, fontweight="bold")

    # 1 Input
    stage(0.1, 2.6, 1.9, 1.6, "1. Input",
          formula="Query $q$,\ncontext $\\{c_1..c_n\\}$",
          color="#FFF7E0")
    # 2 Graph build
    stage(2.3, 2.6, 2.0, 1.6, "2. Build Graph",
          formula="$w_{ij}=\\cos(\\phi_i,\\phi_j)$\n$\\geq \\tau$",
          color="#EAF1FB")
    # 3 Inject current
    stage(4.6, 2.6, 1.9, 1.6, "3. Inject Current",
          formula="$I_i = \\cos(q, c_i)$",
          color="#EAF1FB")
    # 4 Solve potentials
    stage(6.8, 2.6, 2.0, 1.6, "4. Solve Kirchhoff",
          formula="$L\\,p = I$\n(sparse CG)",
          color="#F0E5F2")
    # 5 Update conductance
    stage(9.1, 2.6, 1.8, 1.6, "5. Reinforce",
          formula="$D \\leftarrow (1{-}\\mu)D + \\alpha|Q|^{\\gamma}$",
          color="#F0E5F2")

    # forward arrows
    arrow(2.0, 3.4, 2.3, 3.4)
    arrow(4.3, 3.4, 4.6, 3.4)
    arrow(6.5, 3.4, 6.8, 3.4)
    arrow(8.8, 3.4, 9.1, 3.4)

    # 6 Convergence check (below stage 5)
    stage(8.6, 0.4, 2.3, 1.5, "6. Convergence?",
          formula="$\\sum|\\Delta D| < \\varepsilon \\sum D$",
          color="#FFE9E6")

    # loop-back arrow (red) curving from convergence -> step 3 (re-inject)
    # use distinct vertical lanes so red "no" and green "yes" never overlap
    arrow(8.6, 1.45, 5.55, 1.45, color=C_ACCENT)
    ax.text(7.0, 1.6, "iterate (no)", ha="center", fontsize=8.5,
            color=C_ACCENT, fontweight="bold")
    arrow(5.55, 1.45, 5.55, 2.6, color=C_ACCENT)

    # 7 Prune (left)
    stage(0.1, 0.4, 2.0, 1.5, "7. Prune Top-k",
          formula="rank by\nnode flow $F_i$",
          color="#E6F4EA")
    # yes branch on a LOWER lane
    arrow(8.6, 0.75, 2.1, 0.75, color=C_OK)
    ax.text(5.0, 0.92, "yes (converged)", ha="center", fontsize=8.5,
            color=C_OK, fontweight="bold")

    # Legend
    leg = [
        Line2D([0], [0], color=C_PRIMARY, lw=2, label="Forward pass"),
        Line2D([0], [0], color=C_ACCENT, lw=2, label="Iterate (Kirchhoff loop)"),
        Line2D([0], [0], color=C_OK, lw=2, label="Output"),
    ]
    ax.legend(handles=leg, loc="lower right", bbox_to_anchor=(1.0, -0.02),
              ncol=3, frameon=False, fontsize=9)

    save(fig, "fig2_algorithm_flow")


# ---------------------------------------------------------------------------
# Figure 3 : Novelty positioning (axes: granularity vs interaction modeling)
# ---------------------------------------------------------------------------
def fig_novelty():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.set_title("Positioning: FCNP vs. Prior Context-Compression Methods", loc="left", pad=10)

    # Axes
    ax.annotate("", xy=(9.4, 0.6), xytext=(0.6, 0.6),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_EDGE))
    ax.annotate("", xy=(0.6, 5.9), xytext=(0.6, 0.6),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color=C_EDGE))
    ax.text(5.0, 0.25, "Granularity (per-item $\\rightarrow$ structural)",
            ha="center", fontsize=10, fontweight="bold", color=C_EDGE)
    ax.text(0.25, 3.25, "Inter-element interaction modeling",
            ha="center", va="center", rotation=90,
            fontsize=10, fontweight="bold", color=C_EDGE)

    # Quadrant labels removed - the x/y axis labels already convey this.

    # Hide spines and ticks
    for s in ["top", "right", "left", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Place methods - label sits BELOW circle with consistent padding,
    # family italic sits below the label.
    def method(x, y, label, family, color, size=900, r=0.20):
        ax.scatter([x], [y], s=size, c=color, edgecolors=C_EDGE,
                   linewidths=1.3, zorder=3)
        # offset chosen to clear the circle radius
        ax.text(x, y - r - 0.30, label, ha="center", fontsize=8.5,
                fontweight="bold", color=C_EDGE)
        ax.text(x, y - r - 0.55, family, ha="center", fontsize=7.5,
                color=C_MUTED, style="italic")

    # Per-item, low interaction (bottom-left)
    method(1.6, 2.0, "BM25", "lexical-rank", "#B0BEC5")
    method(3.4, 2.5, "Dense Top-k", "embedding-rank", "#B0BEC5")
    # Per-item with stronger info-theoretic signal (middle band)
    method(2.4, 4.3, "Selective Context", "info-theoretic", "#90A4AE")
    method(4.4, 4.9, "LLMLingua / -2", "prompt-comp.", "#90A4AE")
    method(5.0, 3.5, "LongLLMLingua", "prompt-comp.", "#90A4AE")
    method(5.8, 4.4, "RECOMP", "extractive/abstr.", "#90A4AE")

    # FCNP (top-right) — highlighted
    method(8.2, 5.3, "FCNP (this work)", "graph-flow", C_ACCENT, size=1500, r=0.30)

    # Dashed boundary showing the "gap" FCNP fills
    ax.plot([6.7, 6.7], [0.8, 5.9], "--", color=C_MUTED, lw=1, alpha=0.5)
    ax.text(3.7, 0.95, "$\\leftarrow$ per-item ranking",
            fontsize=9, color=C_MUTED, fontweight="bold", ha="center")
    ax.text(8.05, 0.95, "global flow optimization $\\rightarrow$",
            fontsize=9, color=C_ACCENT, fontweight="bold", ha="center")

    save(fig, "fig3_novelty")


# ---------------------------------------------------------------------------
# Figure 4 : Context graph before / after pruning
# ---------------------------------------------------------------------------
def fig_context_graph():
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
    fig.suptitle("Context Graph: Before and After FCNP Pruning",
                 fontsize=12, fontweight="bold", y=1.02)

    # Layout: ring of context nodes + source (top) + sink (bottom)
    n = 10
    angles = np.linspace(np.pi*0.95, np.pi*0.05, n)
    radius = 1.6
    cx, cy = 0.0, 0.0
    nodes = np.array([[cx + radius*np.cos(a), cy + radius*np.sin(a)] for a in angles])
    src = np.array([0.0, 2.4])
    snk = np.array([0.0, -2.4])

    # Edges (random sparse) + ground-truth high-flow chain
    base_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if rng.random() < 0.32:
                base_edges.add((i, j))
    # Highlight chain that will survive pruning: 1 -> 4 -> 7
    chain = [(1, 4), (4, 7)]
    for e in chain:
        base_edges.add(e)
    # connect src/snk
    src_edges = {0, 1, 2, 3}
    snk_edges = {6, 7, 8, 9}

    def draw_panel(ax, retained=None, title=""):
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(-3.0, 3.0)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", color=C_EDGE)
        # Source / sink
        ax.add_patch(Circle(src, 0.28, color=C_PRIMARY, zorder=3))
        ax.text(src[0], src[1]+0.05, "q", color="white", ha="center", va="center",
                fontsize=11, fontweight="bold", zorder=4)
        ax.text(src[0], src[1]+0.55, "query (source)", ha="center", fontsize=8.5,
                color=C_PRIMARY)
        ax.add_patch(Circle(snk, 0.28, color=C_ACCENT, zorder=3))
        ax.text(snk[0], snk[1], "*", color="white", ha="center", va="center",
                fontsize=14, fontweight="bold", zorder=4)
        ax.text(snk[0], snk[1]-0.55, "answer (sink)", ha="center", fontsize=8.5,
                color=C_ACCENT)

        # Inter-node edges
        for (i, j) in base_edges:
            keep = retained is None or (i in retained and j in retained)
            color = C_PRIMARY if keep and retained is not None else "#B0BEC5"
            lw = 2.2 if keep and retained is not None else 0.7
            alpha = 1.0 if keep else 0.25
            ax.plot([nodes[i, 0], nodes[j, 0]], [nodes[i, 1], nodes[j, 1]],
                    "-", color=color, lw=lw, alpha=alpha, zorder=1)
        # src / snk edges
        for i in src_edges:
            keep = retained is None or i in retained
            color = C_PRIMARY if keep and retained is not None else "#B0BEC5"
            ax.plot([src[0], nodes[i, 0]], [src[1], nodes[i, 1]],
                    "-", color=color, lw=2.0 if keep and retained is not None else 0.6,
                    alpha=1.0 if keep else 0.22, zorder=1)
        for i in snk_edges:
            keep = retained is None or i in retained
            color = C_ACCENT if keep and retained is not None else "#B0BEC5"
            ax.plot([snk[0], nodes[i, 0]], [snk[1], nodes[i, 1]],
                    "-", color=color, lw=2.0 if keep and retained is not None else 0.6,
                    alpha=1.0 if keep else 0.22, zorder=1)

        # Context nodes
        for i, (x, y) in enumerate(nodes):
            keep = retained is None or i in retained
            color = C_HIGHLIGHT if keep and retained is not None else "#E0E0E0"
            ax.add_patch(Circle((x, y), 0.20, facecolor=color,
                                edgecolor=C_EDGE, lw=1.0,
                                alpha=1.0 if keep else 0.35, zorder=2))
            ax.text(x, y, f"$c_{{{i}}}$", ha="center", va="center",
                    fontsize=7.5, color=C_EDGE,
                    alpha=1.0 if keep else 0.35, zorder=3)

    draw_panel(axes[0], retained=None, title="(a) Initial: $|V|=10$ context items")
    # After pruning: keep nodes on the high-flow path
    retained = {1, 3, 4, 7, 8}
    draw_panel(axes[1], retained=retained,
               title="(b) FCNP-pruned: top-$k$ by node flow $F_i$")

    # Annotation
    axes[1].annotate("high-flow path\n(retained)", xy=(0.4, -0.1), xytext=(1.4, -1.4),
                     fontsize=8.5, color=C_ACCENT, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))

    save(fig, "fig4_context_graph")


if __name__ == "__main__":
    fig_architecture()
    fig_algorithm_flow()
    fig_novelty()
    fig_context_graph()
    print("done")
