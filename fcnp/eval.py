"""Evaluation harness: retrieval metrics, statistical tests, aggregation.

Computes per-example metrics for every compression method on every
ToolBench example, then aggregates with bootstrap 95% confidence
intervals and paired significance tests (Wilcoxon signed-rank).

Metrics
-------
- Recall@k      : |relevant ∩ kept| / |relevant|
- Precision@k   : |relevant ∩ kept| / k
- F1@k          : harmonic mean
- nDCG@k        : normalized discounted cumulative gain (assumes the
                  ranking order produced by each method, with binary
                  relevance from ground truth)
- CompressionRatio
- ReductionPct
- LatencyMs
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np

from fcnp.baselines.base import BaselineResult
from fcnp.baselines.fcnp_wrapper import FCNPMethod
from fcnp.datasets.toolbench import ToolBenchExample
from fcnp.types import ContextElement


# ---------------------------------------------------------------------
# Per-example metric primitives
# ---------------------------------------------------------------------
def recall_at_k(survivors: list[ContextElement], relevant_keys: set[str]) -> float:
    if not relevant_keys:
        return 1.0
    kept = {e.id for e in survivors}
    return len(kept & relevant_keys) / len(relevant_keys)


def precision_at_k(survivors: list[ContextElement], relevant_keys: set[str]) -> float:
    if not survivors:
        return 0.0
    kept = {e.id for e in survivors}
    return len(kept & relevant_keys) / len(survivors)


def f1_at_k(survivors, relevant_keys) -> float:
    r = recall_at_k(survivors, relevant_keys)
    p = precision_at_k(survivors, relevant_keys)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(survivors: list[ContextElement], relevant_keys: set[str]) -> float:
    if not relevant_keys or not survivors:
        return 0.0
    rels = [1.0 if e.id in relevant_keys else 0.0 for e in survivors]
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))
    ideal_rels = sorted(
        [1.0] * min(len(relevant_keys), len(survivors))
        + [0.0] * max(0, len(survivors) - len(relevant_keys)),
        reverse=True,
    )
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------
# Per-example record
# ---------------------------------------------------------------------
@dataclass
class ExampleScore:
    method: str
    split: str
    query_id: str
    n_input: int
    n_output: int
    input_tokens: int
    output_tokens: int
    compression_ratio: float
    reduction_pct: float
    recall: float
    precision: float
    f1: float
    ndcg: float
    latency_ms: float


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------
def run_example(
    method,
    example: ToolBenchExample,
    keep_k: int,
    embedder=None,
) -> ExampleScore:
    elements = example.to_elements(embedder=embedder)
    relevant = example.relevant_keys()

    # Build query embedding from same embedder for parity
    if embedder:
        q_emb = embedder(example.query)
    else:
        from fcnp.datasets.toolbench import _hash_embedding
        q_emb = _hash_embedding(example.query, dim=elements[0].embedding.shape[0]) if elements else None

    res: BaselineResult = method.compress(
        elements=elements,
        query_text=example.query,
        query_embedding=q_emb,
        keep_k=keep_k,
    )
    return ExampleScore(
        method=method.name,
        split=example.split,
        query_id=example.query_id,
        n_input=res.n_input,
        n_output=res.n_output,
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        compression_ratio=res.compression_ratio,
        reduction_pct=res.reduction_pct,
        recall=recall_at_k(res.survivors, relevant),
        precision=precision_at_k(res.survivors, relevant),
        f1=f1_at_k(res.survivors, relevant),
        ndcg=ndcg_at_k(res.survivors, relevant),
        latency_ms=res.wall_time_ms,
    )


def evaluate_all(
    methods: dict[str, "object"],
    examples: dict[str, list[ToolBenchExample]],
    keep_k_strategy: str = "oracle",
    fixed_k: int = 10,
    embedder=None,
) -> list[ExampleScore]:
    """Run every method on every example.

    keep_k_strategy
    ---------------
    - "oracle"   : k = |relevant_apis| (compare methods at equal budget)
    - "fixed"    : k = fixed_k
    - "fraction" : k = max(1, ceil(0.1 * |api_list|))
    """
    scores: list[ExampleScore] = []
    for split, exs in examples.items():
        for ex in exs:
            if not ex.api_list:
                continue
            if keep_k_strategy == "oracle":
                k = max(1, len(ex.relevant_keys()))
            elif keep_k_strategy == "fixed":
                k = fixed_k
            elif keep_k_strategy == "fraction":
                k = max(1, math.ceil(0.10 * len(ex.api_list)))
            else:
                raise ValueError(keep_k_strategy)
            for name, m in methods.items():
                scores.append(run_example(m, ex, keep_k=k, embedder=embedder))
    return scores


# ---------------------------------------------------------------------
# Aggregation + statistics
# ---------------------------------------------------------------------
def bootstrap_ci(values: list[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    means = np.empty(n_boot)
    n = arr.size
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = arr[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(arr.mean()), lo, hi


def wilcoxon_paired(a: list[float], b: list[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired samples."""
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return {"p_value": None, "statistic": None, "n": len(a), "note": "scipy missing"}
    a = np.asarray(a); b = np.asarray(b)
    if a.size != b.size or a.size == 0:
        return {"p_value": None, "statistic": None, "n": 0}
    diff = a - b
    if not np.any(diff != 0):
        return {"p_value": 1.0, "statistic": 0.0, "n": a.size}
    stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    return {"p_value": float(p), "statistic": float(stat), "n": int(a.size)}


@dataclass
class MethodAggregate:
    method: str
    n: int
    recall_mean: float; recall_ci_lo: float; recall_ci_hi: float
    precision_mean: float; precision_ci_lo: float; precision_ci_hi: float
    f1_mean: float; f1_ci_lo: float; f1_ci_hi: float
    ndcg_mean: float; ndcg_ci_lo: float; ndcg_ci_hi: float
    compression_ratio_mean: float
    reduction_pct_mean: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float


def aggregate(scores: list[ExampleScore]) -> list[MethodAggregate]:
    by_method: dict[str, list[ExampleScore]] = {}
    for s in scores:
        by_method.setdefault(s.method, []).append(s)

    out: list[MethodAggregate] = []
    for method, rows in by_method.items():
        r = [x.recall for x in rows]
        p = [x.precision for x in rows]
        f = [x.f1 for x in rows]
        nd = [x.ndcg for x in rows]
        cr = [x.compression_ratio for x in rows if math.isfinite(x.compression_ratio)]
        rp = [x.reduction_pct for x in rows]
        lat = [x.latency_ms for x in rows]
        rm, rlo, rhi = bootstrap_ci(r)
        pm, plo, phi = bootstrap_ci(p)
        fm, flo, fhi = bootstrap_ci(f)
        nm, nlo, nhi = bootstrap_ci(nd)
        out.append(MethodAggregate(
            method=method, n=len(rows),
            recall_mean=rm, recall_ci_lo=rlo, recall_ci_hi=rhi,
            precision_mean=pm, precision_ci_lo=plo, precision_ci_hi=phi,
            f1_mean=fm, f1_ci_lo=flo, f1_ci_hi=fhi,
            ndcg_mean=nm, ndcg_ci_lo=nlo, ndcg_ci_hi=nhi,
            compression_ratio_mean=float(np.mean(cr)) if cr else float("inf"),
            reduction_pct_mean=float(np.mean(rp)),
            latency_ms_mean=float(np.mean(lat)),
            latency_ms_p50=float(np.percentile(lat, 50)),
            latency_ms_p95=float(np.percentile(lat, 95)),
        ))
    out.sort(key=lambda m: -m.f1_mean)
    return out


def pairwise_significance(
    scores: list[ExampleScore],
    primary_method: str,
    metric: str = "f1",
) -> dict[str, dict]:
    """Wilcoxon paired tests: primary_method vs each other method.

    Pairs are formed by (split, query_id).
    """
    by_method: dict[str, dict[tuple[str, str], float]] = {}
    for s in scores:
        key = (s.split, s.query_id)
        by_method.setdefault(s.method, {})[key] = getattr(s, metric)

    if primary_method not in by_method:
        return {}
    primary = by_method[primary_method]
    out: dict[str, dict] = {}
    for m, vals in by_method.items():
        if m == primary_method:
            continue
        common = sorted(set(primary) & set(vals))
        a = [primary[k] for k in common]
        b = [vals[k] for k in common]
        out[m] = wilcoxon_paired(a, b)
    return out
