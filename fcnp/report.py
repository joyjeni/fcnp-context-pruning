"""Publication-ready report writer.

Emits:
    - results/metrics.json   : machine-readable, consumed by the
                                Vercel dashboard
    - results/results.csv    : per-example raw scores
    - results/summary.md     : Markdown summary tables
    - figures/*.png          : (optional) matplotlib plots
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

from fcnp.eval import ExampleScore, MethodAggregate, pairwise_significance


def write_report(
    out_dir: str | Path,
    per_example: list[ExampleScore],
    aggregates: list[MethodAggregate],
    primary_method: str = "FCNP",
    dataset_name: str = "ToolBench",
    config_used: dict | None = None,
    run_id: str | None = None,
) -> dict[str, Path]:
    out = Path(out_dir)
    (out / "results").mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    significance = pairwise_significance(per_example, primary_method, metric="f1")

    # ---------------- metrics.json ----------------
    payload = {
        "run_id": run_id or f"run-{int(time.time())}",
        "timestamp": time.time(),
        "dataset": dataset_name,
        "primary_method": primary_method,
        "config": config_used or {},
        "n_examples": len({(s.split, s.query_id) for s in per_example}),
        "methods": [asdict(m) for m in aggregates],
        "significance_vs_primary": {
            "metric": "f1",
            "tests": significance,
        },
        "per_split_summary": _per_split(per_example),
    }
    p = out / "results" / "metrics.json"
    p.write_text(json.dumps(payload, indent=2, default=float))
    paths["metrics_json"] = p

    # ---------------- results.csv ----------------
    p = out / "results" / "results.csv"
    if per_example:
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(per_example[0]).keys()))
            w.writeheader()
            for s in per_example:
                w.writerow(asdict(s))
    paths["results_csv"] = p

    # ---------------- summary.md ----------------
    p = out / "results" / "summary.md"
    p.write_text(_render_markdown(aggregates, significance, primary_method, dataset_name))
    paths["summary_md"] = p

    return paths


def _per_split(per_example: list[ExampleScore]) -> dict[str, dict[str, dict[str, float]]]:
    """split -> method -> mean of each metric."""
    out: dict = {}
    by = {}
    for s in per_example:
        by.setdefault(s.split, {}).setdefault(s.method, []).append(s)
    for split, by_method in by.items():
        out[split] = {}
        for m, rows in by_method.items():
            n = len(rows)
            out[split][m] = {
                "n": n,
                "recall": sum(r.recall for r in rows) / n,
                "precision": sum(r.precision for r in rows) / n,
                "f1": sum(r.f1 for r in rows) / n,
                "ndcg": sum(r.ndcg for r in rows) / n,
                "compression_ratio": sum(
                    r.compression_ratio for r in rows if r.compression_ratio != float("inf")
                ) / max(1, sum(1 for r in rows if r.compression_ratio != float("inf"))),
                "latency_ms": sum(r.latency_ms for r in rows) / n,
            }
    return out


def _render_markdown(
    aggregates: list[MethodAggregate],
    significance: dict,
    primary: str,
    dataset: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# FCNP results on {dataset}\n")
    lines.append("Primary method: **{}**\n".format(primary))
    lines.append("## Main results (mean ± 95% bootstrap CI)\n")
    lines.append(
        "| Method | n | Recall | Precision | F1 | nDCG | Compression × | Reduction % | Latency p50 (ms) | Latency p95 (ms) |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for m in aggregates:
        lines.append(
            f"| {m.method} | {m.n} "
            f"| {m.recall_mean:.3f} [{m.recall_ci_lo:.3f}, {m.recall_ci_hi:.3f}] "
            f"| {m.precision_mean:.3f} [{m.precision_ci_lo:.3f}, {m.precision_ci_hi:.3f}] "
            f"| {m.f1_mean:.3f} [{m.f1_ci_lo:.3f}, {m.f1_ci_hi:.3f}] "
            f"| {m.ndcg_mean:.3f} [{m.ndcg_ci_lo:.3f}, {m.ndcg_ci_hi:.3f}] "
            f"| {m.compression_ratio_mean:.2f} "
            f"| {m.reduction_pct_mean:.1f} "
            f"| {m.latency_ms_p50:.2f} "
            f"| {m.latency_ms_p95:.2f} |"
        )
    lines.append("")
    lines.append(f"## Pairwise Wilcoxon signed-rank tests (F1, vs {primary})\n")
    lines.append("| Method | n pairs | statistic | p-value | significant (p<0.05) |")
    lines.append("|---|---:|---:|---:|:---:|")
    for m, t in significance.items():
        pv = t.get("p_value")
        sig = "✓" if (pv is not None and pv < 0.05) else ""
        lines.append(
            f"| {m} | {t.get('n', '')} "
            f"| {t.get('statistic', '')} "
            f"| {pv if pv is not None else 'n/a'} | {sig} |"
        )
    lines.append("")
    return "\n".join(lines)
