"""Builds notebooks/fcnp_toolbench_benchmark.ipynb programmatically.

Kept as a build script so the notebook stays in sync with the package
and can be regenerated. Run:

    python notebooks/build_notebook.py
"""

import json
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(*lines):
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": [l + "\n" for l in lines],
    }


CELLS = [
    md(
        "# FCNP on ToolBench — Publication Benchmark",
        "",
        "**Flow-Based Context Network Pruning (FCNP)** compared against six SOTA / reference baselines on the OpenBMB **ToolBench** benchmark.",
        "",
        "Pipeline:",
        "",
        "1. Load ToolBench benchmark splits (G1/G2/G3) from HuggingFace.",
        "2. Embed each candidate API document with `sentence-transformers/all-MiniLM-L6-v2`.",
        "3. Run FCNP and 7 baselines at matched compression budget (oracle k).",
        "4. Compute Recall@k / Precision@k / F1@k / nDCG@k / compression × / latency.",
        "5. Aggregate with bootstrap 95% CI; paired Wilcoxon signed-rank tests vs FCNP.",
        "6. Emit `metrics.json` and POST to the Vercel dashboard.",
        "",
        "GitHub: https://github.com/joyjeni/fcnp-context-pruning",
    ),
    md("## 1. Environment setup"),
    code(
        "!pip install -q sentence-transformers datasets scipy",
        "!pip install -q git+https://github.com/joyjeni/fcnp-context-pruning.git",
        "",
        "import os, json, time, math",
        "import numpy as np",
        "from pathlib import Path",
    ),
    md("## 2. Configuration"),
    code(
        "# ---- Experiment configuration ----",
        "SPLITS = ['g1_instruction', 'g1_category', 'g1_tool',",
        "          'g2_instruction', 'g2_category', 'g3_instruction']",
        "LIMIT_PER_SPLIT = 200    # ToolBench provides 200 test instances per split",
        "EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'",
        "OUT_DIR = Path('/kaggle/working')",
        "",
        "# ---- Vercel dashboard endpoint (set as Kaggle secrets) ----",
        "DASHBOARD_URL   = os.environ.get('DASHBOARD_URL', '')   # e.g. https://fcnp.vercel.app",
        "DASHBOARD_TOKEN = os.environ.get('DASHBOARD_TOKEN', '')",
    ),
    md("## 3. Load ToolBench"),
    code(
        "from fcnp.datasets.toolbench import ToolBenchLoader",
        "",
        "loader = ToolBenchLoader.from_huggingface(",
        "    repo='tuandunghcmut/toolbench-v1', config='benchmark'",
        ")",
        "examples = {}",
        "for s in SPLITS:",
        "    try:",
        "        examples[s] = list(loader.iter_split(s, limit=LIMIT_PER_SPLIT))",
        "        print(f'  {s}: {len(examples[s])} examples '",
        "              f'(avg api_list={np.mean([len(e.api_list) for e in examples[s]]):.1f}, '",
        "              f'avg relevant={np.mean([len(e.relevant_apis) for e in examples[s]]):.2f})')",
        "    except Exception as e:",
        "        print(f'  skip {s}: {e}')",
    ),
    md("## 4. Sentence-transformer embedder"),
    code(
        "from sentence_transformers import SentenceTransformer",
        "import torch",
        "",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        "print('device:', device)",
        "encoder = SentenceTransformer(EMBEDDING_MODEL, device=device)",
        "",
        "def embed(text: str) -> np.ndarray:",
        "    return encoder.encode(text, convert_to_numpy=True,",
        "                          normalize_embeddings=True).astype(np.float32)",
        "",
        "# Batched embedder for speed",
        "class BatchedEmbedder:",
        "    def __init__(self, model, batch_size=64):",
        "        self.model = model; self.batch_size = batch_size",
        "    def __call__(self, text):",
        "        return self.model.encode(text, convert_to_numpy=True,",
        "                                 normalize_embeddings=True).astype(np.float32)",
        "embedder = BatchedEmbedder(encoder)",
    ),
    md("## 5. Configure methods"),
    code(
        "from fcnp import FCNPConfig, FCNPMethod, ALL_BASELINES",
        "",
        "fcnp_cfg = FCNPConfig(",
        "    similarity_threshold=0.25,",
        "    epsilon=1e-3,",
        "    max_iterations=200,",
        "    mu=0.10,",
        "    alpha=0.50,",
        "    gamma=1.20,",
        ")",
        "",
        "methods = {name: cls() for name, cls in ALL_BASELINES.items()}",
        "methods['FCNP'] = FCNPMethod(config=fcnp_cfg)",
        "list(methods.keys())",
    ),
    md("## 6. Run benchmark"),
    code(
        "from fcnp.eval import evaluate_all, aggregate",
        "",
        "t0 = time.perf_counter()",
        "scores = evaluate_all(methods, examples,",
        "                      keep_k_strategy='oracle',",
        "                      embedder=embedder)",
        "print(f'total time: {time.perf_counter() - t0:.1f}s')",
        "print(f'evaluations: {len(scores)}')",
        "",
        "agg = aggregate(scores)",
        "for m in agg:",
        "    print(f'{m.method:<20} F1={m.f1_mean:.3f} '",
        "          f'Recall={m.recall_mean:.3f} nDCG={m.ndcg_mean:.3f} '",
        "          f'Comp×={m.compression_ratio_mean:.2f} '",
        "          f'p95_ms={m.latency_ms_p95:.1f}')",
    ),
    md("## 7. Statistical tests"),
    code(
        "from fcnp.eval import pairwise_significance",
        "",
        "sig = pairwise_significance(scores, primary_method='FCNP', metric='f1')",
        "for m, t in sig.items():",
        "    pv = t.get('p_value')",
        "    flag = '***' if pv is not None and pv < 0.001 else ('**' if pv is not None and pv < 0.01 else ('*' if pv is not None and pv < 0.05 else ''))",
        "    print(f'FCNP vs {m:<20} n={t[\"n\"]} W={t[\"statistic\"]:.1f} p={pv:.4g} {flag}')",
    ),
    md("## 8. Save metrics + per-example CSV"),
    code(
        "from fcnp import write_report",
        "",
        "paths = write_report(",
        "    OUT_DIR, scores, agg,",
        "    primary_method='FCNP',",
        "    dataset_name='ToolBench (G1/G2/G3 benchmark splits)',",
        "    config_used=fcnp_cfg.__dict__,",
        ")",
        "for k, v in paths.items():",
        "    print(k, v)",
    ),
    md("## 9. POST metrics to Vercel dashboard"),
    code(
        "import urllib.request, urllib.error",
        "",
        "if DASHBOARD_URL and DASHBOARD_TOKEN:",
        "    payload = json.loads(paths['metrics_json'].read_text())",
        "    req = urllib.request.Request(",
        "        f'{DASHBOARD_URL.rstrip(\"/\")}/api/metrics',",
        "        data=json.dumps(payload).encode(),",
        "        headers={",
        "            'Content-Type': 'application/json',",
        "            'Authorization': f'Bearer {DASHBOARD_TOKEN}',",
        "        }, method='POST',",
        "    )",
        "    try:",
        "        with urllib.request.urlopen(req, timeout=30) as r:",
        "            print('dashboard response:', r.status, r.read()[:200])",
        "    except urllib.error.HTTPError as e:",
        "        print('dashboard error:', e.code, e.read()[:200])",
        "else:",
        "    print('DASHBOARD_URL / DASHBOARD_TOKEN not set; metrics saved locally only.')",
    ),
    md(
        "## 10. Plots",
        "",
        "Pareto frontier (F1 vs compression ratio) and latency comparison.",
    ),
    code(
        "import matplotlib.pyplot as plt",
        "",
        "fig, axes = plt.subplots(1, 2, figsize=(13, 5))",
        "for m in agg:",
        "    axes[0].errorbar(",
        "        m.compression_ratio_mean, m.f1_mean,",
        "        yerr=[[m.f1_mean - m.f1_ci_lo], [m.f1_ci_hi - m.f1_mean]],",
        "        fmt='o', label=m.method, markersize=8, capsize=4,",
        "    )",
        "    axes[0].annotate(m.method,",
        "                     (m.compression_ratio_mean, m.f1_mean),",
        "                     textcoords='offset points', xytext=(6, 6), fontsize=9)",
        "axes[0].set_xlabel('Compression ratio (input tokens / output tokens)')",
        "axes[0].set_ylabel('F1@k (oracle budget)')",
        "axes[0].set_title('Pareto: accuracy vs compression')",
        "axes[0].grid(alpha=0.3)",
        "",
        "names = [m.method for m in agg]",
        "lat = [m.latency_ms_p50 for m in agg]",
        "axes[1].barh(names, lat)",
        "axes[1].set_xlabel('Latency p50 (ms / query)')",
        "axes[1].set_title('Per-query compression latency')",
        "axes[1].set_xscale('log')",
        "plt.tight_layout()",
        "plt.savefig(OUT_DIR / 'figures_pareto.png', dpi=150, bbox_inches='tight')",
        "plt.show()",
    ),
    md(
        "## 11. Publication summary",
        "",
        "The Markdown summary in `results/summary.md` is the camera-ready",
        "table for the paper. Per-example raw scores are in `results.csv`",
        "for reviewer-supplied re-analysis.",
    ),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    out = Path(__file__).parent / "fcnp_toolbench_benchmark.ipynb"
    out.write_text(json.dumps(NB, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
