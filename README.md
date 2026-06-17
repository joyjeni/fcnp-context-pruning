# FCNP — Flow-Coupled Network Pruning

> **PhD Objective 4** | Kirchhoff/Physarum-Analog Context Compression for LLM Agents

[![GitHub](https://img.shields.io/badge/GitHub-Public-black)](https://github.com/joyjeni/fcnp-context-pruning)
[![Vercel](https://img.shields.io/badge/Vercel-Dashboard-brightgreen)](https://fcnp-context-pruning.vercel.app)
[![Kaggle](https://img.shields.io/badge/Kaggle-Benchmark-blue)](https://www.kaggle.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**GitHub**: https://github.com/joyjeni/fcnp-context-pruning (PUBLIC — previously private, now open)

---

## What is FCNP?

FCNP (**Flow-Coupled Network Pruning**) is a context compression algorithm for LLM agents inspired by two physical systems:

1. **Kirchhoff's circuit laws** — global current conservation constrains which nodes (context chunks) carry significant flow.
2. **Physarum polycephalum (slime mould)** — a biological network that self-organises to find shortest paths by reinforcing high-flow tubes and pruning low-flow ones.

FCNP models the context window as a conductance network: each chunk of retrieved context is a node; query-chunk relevance defines conductance; Kirchhoff flow equations determine which chunks are "on the critical path" to answering the query. Low-flow chunks are pruned.

**Novel Contribution**: FCNP is the **first global flow-coupled context compression method for LLMs** — prior work (SelectiveContext, LLMLingua) performs per-item ranking independently of other items. FCNP's Kirchhoff formulation makes pruning decisions jointly, preserving globally coherent context rather than locally top-ranked fragments.

---

## Deployment

**GitHub Repository**: https://github.com/joyjeni/fcnp-context-pruning

**Vercel Dashboard**: deployed from the `dashboard/` directory of the repository. Provides live compression visualisation — shows the conductance graph, flow values, and which chunks are pruned vs retained.

### Kaggle Notebook

`fcnp_toolbench_benchmark.ipynb` — full reproducible benchmark against 7 baselines on the ToolBench-Agri dataset.

---

## Core Equations

### Kirchhoff Flow System

FCNP solves the following linear system to compute flow through each context chunk:

$$L(D) \cdot \mathbf{p} = \mathbf{I}$$

where:
- $L(D)$ — the **Laplacian** of the conductance graph $D$ (chunk-to-chunk conductance matrix)
- $\mathbf{p}$ — pressure vector (one value per context chunk)
- $\mathbf{I}$ — injected current vector (query relevance signal, computed from embeddings)

Solving for $\mathbf{p}$ yields pressure differences across edges; flow on each edge is $f_{ij} = D_{ij} \cdot (p_i - p_j)$. Chunks on high-flow edges are retained; low-flow chunks are pruned.

### Conductance Update (Physarum Dynamics)

Conductances are updated iteratively to reinforce high-flow paths (physarum-inspired adaptive network):

$$D_{ij}(t+1) = (1 - \mu) \cdot D_{ij}(t) + \alpha \cdot |Q_{ij}|^\gamma$$

where:
- $D_{ij}(t)$ — conductance on edge $(i,j)$ at iteration $t$
- $Q_{ij}$ — flow through edge $(i,j)$ (from Kirchhoff solution at step $t$)
- $\mu$ — decay rate (prunes edges that carry little flow)
- $\alpha$ — growth rate (reinforces high-flow edges)
- $\gamma > 1$ — superlinear exponent (amplifies differences, accelerates convergence)

### Convergence Criterion

Iteration terminates when the conductance matrix change falls below tolerance:

$$\frac{\|D(t+1) - D(t)\|_F}{\|D(t)\|_F} < \epsilon$$

In practice, convergence is reached in 8–15 iterations for typical mandi price response tables.

---

## Results

### Benchmark: ToolBench-Agri (Agricultural API responses)

| Metric               | Value          | Notes                                          |
|----------------------|----------------|------------------------------------------------|
| Compression Ratio    | **10:1**       | 50+ records → ~5 retained chunks              |
| F1@K                 | **Best**       | Outperforms all 7 baselines (Wilcoxon p<0.05) |
| Citation Accuracy    | **≥99%**       | Retained chunks preserve source attribution   |

**Statistical test**: Wilcoxon signed-rank, p < 0.05 across all 7 baseline comparisons.

### Baselines Defeated

| # | Baseline                                            | Reference                    |
|---|-----------------------------------------------------|------------------------------|
| 1 | NoCompression                                       | —                            |
| 2 | Random                                              | —                            |
| 3 | TopKImportance                                      | —                            |
| 4 | BM25                                                | Robertson & Zaragoza, 2009   |
| 5 | DenseTopK                                           | —                            |
| 6 | SelectiveContext                                    | Li et al., EMNLP 2023        |
| 7 | LLMLingua                                           | Jiang et al., EMNLP 2023     |

FCNP produces higher F1@K than all seven baselines while maintaining ≥99% citation accuracy (the retained chunks always include the correct source record).

---

## Data Sources

### data.gov.in — Mandi Price Tables

The primary use case that motivates FCNP's 10:1 compression target: a single data.gov.in API call for commodity prices may return **50–100+ records** (all mandis in a state for a given commodity). This exceeds a typical LLM's useful context budget.

FCNP prunes the response to the **top-5 most relevant mandi records** for the farmer's query, preserving:
- The farmer's district/state preference (high conductance to local mandis)
- The query commodity (high conductance to matching commodity records)
- Recency (high conductance to today's prices)

This makes the compressed context small enough to fit in the LLM's prompt while retaining all citation-worthy data.

---

## Algorithm Overview

```
Input:  context chunks C = {c_1, ..., c_n}, query q
Output: pruned context C' ⊆ C, |C'| << |C|

1. Embed all chunks and query using google/embeddinggemma-300m
2. Build conductance matrix D_0:
      D_ij = cosine_sim(embed(c_i), embed(c_j)) * query_relevance(c_i, q)
3. Inject current I_i = cosine_sim(embed(c_i), embed(q))
4. Iterate until convergence:
      a. Solve L(D) · p = I  (sparse linear solve)
      b. Compute flows Q_ij = D_ij · |p_i - p_j|
      c. Update D_ij(t+1) = (1−μ)·D_ij + α·|Q_ij|^γ
5. Select top-K chunks by total incident flow: Σ_j Q_ij
6. Return C' = top-K chunks + source attributions
```

---

## Repository Structure

```
fcnp-context-pruning/
├── src/
│   ├── fcnp.py              # Core algorithm: Kirchhoff solve + physarum update
│   ├── conductance.py       # Build D matrix from embeddings
│   ├── laplacian.py         # L(D) construction and sparse solve
│   ├── embedding.py         # google/embeddinggemma-300m wrapper
│   └── pruner.py            # Top-K selection, citation preservation
├── experiments/
│   └── fcnp_toolbench_benchmark.py   # All 7 baselines, F1@K, citation accuracy
├── kaggle/
│   └── fcnp_toolbench_benchmark.ipynb
├── dashboard/               # Vercel deployment: conductance graph visualisation
│   ├── src/
│   │   └── App.jsx          # React conductance graph + flow visualiser
│   └── package.json
└── docs/
    └── README_OBJ4.md       # This file
```

---

## Integration with PhD Pipeline

FCNP is **Objective 4** in the four-component PhD pipeline and completes the loop:

```
[Obj3: MNCD] ──mesh context──► [Obj4: FCNP] ──pruned context──► [Obj1: SessionRerank+]
                                     ▲
                         OctoRoute arm label
                          (from Obj2/APRR)
                          gates domain pruning
```

### Incoming Signals

| Source       | Signal                      | Usage in FCNP                                          |
|--------------|-----------------------------|--------------------------------------------------------|
| Obj3/MNCD    | Full mesh context window    | The raw context to compress (50+ records → top-5)      |
| Obj2/APRR    | OctoRoute arm label `<octo_k>` | Gates domain-specific pruning strategy (price vs. advisory vs. market) |

### Outgoing Signals

| Destination         | Signal                      | Purpose                                                |
|---------------------|-----------------------------|--------------------------------------------------------|
| Obj1/SessionRerank+ | Pruned context + hit/miss   | Co-activation cache update (success reinforces edge)   |

### OctoRoute Gating

The `<octo_k>` arm label from APRR/OctoRoute selects which domain-specific pruning parameters to apply:

```python
DOMAIN_PARAMS = {
    "price":    {"gamma": 1.5, "top_k": 5},   # price queries → tight compression
    "market":   {"gamma": 1.2, "top_k": 8},   # market listings → slightly more context
    "advisory": {"gamma": 1.1, "top_k": 10},  # crop advisory → retain more context
}

def get_params(octo_arm: str) -> dict:
    return DOMAIN_PARAMS.get(octo_arm, {"gamma": 1.3, "top_k": 5})
```

---

## Multilingual Support

FCNP is **language-agnostic** — it operates entirely on dense embeddings (`google/embeddinggemma-300m`) computed from English text (post-translation). The conductance matrix, Kirchhoff flow solve, and physarum update contain no language-specific logic.

No changes are required for multilingual support. The translation layer in Obj1/SessionRerank+ handles language detection and translation before context reaches FCNP.

See [`/docs/multilingual_integration.md`](./multilingual_integration.md) for full multilingual design.

---

## Running Locally

```bash
git clone https://github.com/joyjeni/fcnp-context-pruning
cd fcnp-context-pruning
pip install -r requirements.txt  # torch, transformers, scipy, numpy

# Run benchmark vs all 7 baselines
python experiments/fcnp_toolbench_benchmark.py

# Launch dashboard locally
cd dashboard && npm install && npm run dev
```

### Minimal Usage Example

```python
from src.fcnp import FCNPPruner

pruner = FCNPPruner(top_k=5, gamma=1.5, mu=0.1, alpha=0.9)

# context_chunks: list of strings (mandi price records)
# query: farmer's question in English
pruned = pruner.compress(context_chunks, query="What is today's tomato price?")
# Returns: top-5 most relevant mandi records with source attribution
```

---

## Citation

```bibtex
@misc{fcnp2026,
  title  = {FCNP: Flow-Coupled Network Pruning for LLM Context Compression},
  author = {Jeni, Joy},
  year   = {2026},
  note   = {PhD Objective 4. https://github.com/joyjeni/fcnp-context-pruning}
}
```

### Referenced Baselines

- Li et al. *Compressing Context to Enhance Inference Efficiency of Large Language Models*. EMNLP 2023. (SelectiveContext)
- Jiang et al. *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*. EMNLP 2023. (LLMLingua)

---

*Part of the PhD Agricultural AI pipeline. See also: [Obj1 SessionRerank+](./README_OBJ1.md) | [Obj2 APRR](./README_OBJ2.md) | [Obj3 MNCD](./README_OBJ3.md) | [Multilingual Design](./multilingual_integration.md)*
