# Postdoctoral review — O4 — FCNP (context pruning)

Reviewer stance: a postdoctoral researcher preparing this chapter for a Q1 journal (November 2026).
Severity: **Critical** blocks publication · **Major** a reviewer will require it · **Fixed** done in this commit.

| Severity | Finding | Evidence | Action |
|---|---|---|---|
| Critical | FCNP loses to every simple baseline on its own benchmark | `results/summary.md` and `results/summary_with_hybrid.md`: FCNP F1 0.47 (0.11 in another run) vs BM25 / SelectiveContext / LLMLingua 1.000 and DenseTopK 0.49 (n = 30; Wilcoxon vs the lexical baselines p ≈ 9e-7). The README already notes the synthetic set is lexically trivial — which makes it worse that FCNP does not reach the trivial ceiling. | Do not publish the Physarum pruner as the method. Lead with the typed-block knapsack + PI fidelity controller (ACRS master: Φ-floor met on 99.7 % of turns at 47 % token reduction) and keep Physarum as an ablation or drop it. |
| Critical | "LLMLingua" baseline is a re-implementation | `fcnp/baselines/llmlingua.py` is a heuristic token scorer, not Microsoft's LLMLingua / LLMLingua-2. | Use the official `llmlingua` package (LLMLingua-2) and say so; rename the heuristic. |
| Major | Synthetic G1, n = 30 | 30 synthetic sessions cannot support a journal claim (minimum detectable F1 difference ≈ 0.2). | Evaluate on LongBench / ZeroSCROLLS (task accuracy vs tokens) and on real multi-turn tool sessions; ≥ 300 items. |
| Major | Task outcome not measured | Retention F1 of gold spans is a proxy; reviewers will ask whether the downstream answer stays correct. | Report end-task accuracy with a real LLM reading the pruned context. |

## What this chapter needs to be journal-ready

1. Real LLM agents (open-weight 7–8B class) instead of simulated ones, with ≥ 300 evaluation items per arm.
2. At least two recognised public benchmarks (see the dataset plan in `phd-agentic-ai-master/docs/POSTDOC_ROADMAP.md`).
3. Pre-registered primary metric and one headline statistical test; all other comparisons labelled exploratory.
4. Leak-free splits and a reproducibility command that regenerates every table.
