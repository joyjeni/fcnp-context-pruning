# FCNP — Flow-Based Context Network Pruning

[![tests](https://img.shields.io/badge/tests-passing-3fb950)](./tests)
[![license](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)

A deterministic graph-theoretic algorithm for **LLM context
compression**. Models the context store as a weighted graph; iteratively
reinforces edges proportional to the current they carry under a
Kirchhoff potential field induced by the downstream query; converges to
a sparse subgraph of critical nodes.

> Compared with seven SOTA / reference baselines on **ToolBench**
> (OpenBMB, ICLR 2024) — see `paper/paper.md`.

## Pipeline

```
   ┌────────────────────────┐    metrics.json    ┌──────────────────────┐
   │ Kaggle notebook        │ ─────────────────▶ │ Vercel dashboard     │
   │ fcnp + ToolBench       │  POST /api/metrics │ live tables + charts │
   │ all-MiniLM-L6-v2       │                    │                      │
   └────────────────────────┘                    └──────────────────────┘
```

## Repository layout

```
fcnp/                       Python package (algorithm + baselines + eval)
├── pruner.py               FCNP core
├── baselines/              7 SOTA / reference baselines
├── datasets/toolbench.py   ToolBench loader (HF + local)
├── eval.py                 Recall/Precision/F1/nDCG + bootstrap CI + Wilcoxon
└── report.py               metrics.json + summary.md + CSV

notebooks/                  Kaggle benchmark notebook
dashboard/                  Next.js app deployed on Vercel
paper/paper.md              Publication draft
tests/                      pytest suite
results/                    Pipeline outputs
```

## SOTA baselines included

| Family | Methods |
|---|---|
| Trivial | NoCompression, Random, TopKImportance |
| Retrieval | **BM25**, **DenseTopK** |
| Information-theoretic | **Selective Context** (Li, EMNLP 2023) |
| Prompt compression | **LLMLingua** (Jiang et al., EMNLP 2023) |

## Metrics reported

- Recall@k, Precision@k, F1@k, nDCG@k
- Compression ratio (input/output tokens)
- Reduction percentage
- Per-query latency (mean / p50 / p95)
- Bootstrap 95% confidence intervals ($B = 1000$)
- Paired Wilcoxon signed-rank tests vs FCNP

## Quick start

```bash
pip install -e .
pytest -q

# Validate the pipeline on synthetic data
python run_synthetic_e2e.py
```

## Run the publication benchmark on Kaggle

1. Upload `notebooks/fcnp_toolbench_benchmark.ipynb` to a Kaggle Notebook
   (GPU T4 recommended).
2. Add Kaggle Secrets:
   - `DASHBOARD_URL`   → your Vercel deployment URL
   - `DASHBOARD_TOKEN` → bearer token (matches Vercel env var)
3. Run all cells. The notebook will:
   - load ToolBench benchmark splits via HuggingFace
   - embed with `sentence-transformers/all-MiniLM-L6-v2`
   - run FCNP and 7 baselines at matched oracle budget
   - compute all metrics and significance tests
   - POST `metrics.json` to your dashboard

## Deploy the dashboard

```bash
cd dashboard
npm install
npm run build
vercel --prod
```

Environment variables:

| Variable | Required | Purpose |
|---|---|---|
| `DASHBOARD_TOKEN` | yes | Bearer for `POST /api/metrics` |
| `KV_REST_API_URL` | optional | Vercel KV durable persistence |
| `KV_REST_API_TOKEN` | optional | Vercel KV bearer |

## Configuration reference

| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | 0.30 | Min cosine to create an edge |
| `epsilon` | 1e-4 | Relative convergence threshold |
| `max_iterations` | 200 | Hard iteration cap |
| `mu` | 0.10 | Conductance decay rate |
| `alpha` | 0.50 | Reinforcement gain |
| `gamma` | 1.20 | Flow non-linearity exponent ($>1$) |
| `keep_top_k_fraction` | 0.10 | Node retention budget |

## Citation

```bibtex
@misc{fcnp2026,
  title  = {Flow-Based Context Network Pruning},
  author = {Joyjeni},
  year   = {2026},
  url    = {https://github.com/joyjeni/fcnp-context-pruning}
}
```
