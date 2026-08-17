---
title: FCNP Context Pruning Demo
emoji: 🌐
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# FCNP — Flow-Based Context Network Pruning (interactive demo)

Live demo of **Flow-Based Context Network Pruning (FCNP)**, a Physarum/slime-mould-inspired
([Tero et al. 2010](https://www.science.org/doi/10.1126/science.1177894),
[Bonifaci et al. 2012](https://doi.org/10.1016/j.jtbi.2011.10.021)) graph-flow method for
selecting the most relevant items from a candidate pool (tool APIs, retrieved chunks, etc.)
under a token budget.

Paste a query and a list of candidate documents/APIs (one per line), pick a keep-fraction,
and compare FCNP's selection against BM25, dense top-k cosine, and random baselines side by
side, including latency and the FCNP conductance/flow graph.

Source: [github.com/joyjeni/fcnp-context-pruning](https://github.com/joyjeni/fcnp-context-pruning)

Benchmark results (methodology and honest caveats) are in the project's
[GitHub README](https://github.com/joyjeni/fcnp-context-pruning#readme) and live at the
[Vercel dashboard](https://fcnp-dashboard.vercel.app). On the current benchmark, FCNP does
**not** beat a plain dense top-k baseline on F1 — this demo is for inspecting *how* the
graph-flow selection mechanism behaves, not a claim of state-of-the-art accuracy.
