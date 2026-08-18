"""Reproducible benchmark run adding the FCNP-Hybrid tiering row
(improvements #1-#3) alongside the original synthetic ToolBench-shaped
benchmark, WITHOUT touching `results/summary.md` (the numbers already
cited in the README and reproduced by `run_synthetic_e2e.py`).

Writes to results/summary_with_hybrid.md, results/metrics_with_hybrid.json,
results/results_with_hybrid.csv — a separate, citable artifact for the
recall-vs-compression tradeoff hybrid tiering buys.

    python run_hybrid_comparison.py
"""

import shutil
from pathlib import Path

import numpy as np

from fcnp import (
    ALL_BASELINES, FCNPConfig, FCNPMethod, FCNPHybridMethod,
    evaluate_all, aggregate, write_report,
)
from run_synthetic_e2e import make_example


def main():
    np.random.seed(42)
    examples = {"synthetic_g1": [make_example(s) for s in range(30)]}

    fcnp_cfg = FCNPConfig(
        similarity_threshold=0.25, max_iterations=120, epsilon=1e-3,
        mu=0.10, alpha=0.50, gamma=1.20,
    )
    methods = {name: cls() for name, cls in ALL_BASELINES.items()}
    methods["FCNP"] = FCNPMethod(config=fcnp_cfg)
    methods["FCNP-Hybrid"] = FCNPHybridMethod(config=fcnp_cfg)

    scores = evaluate_all(methods, examples, keep_k_strategy="oracle")
    agg = aggregate(scores)

    tmp_dir = Path("results_hybrid_tmp")
    paths = write_report(
        tmp_dir, scores, agg, primary_method="FCNP",
        dataset_name="ToolBench (synthetic G1) + FCNP-Hybrid row",
        config_used=fcnp_cfg.__dict__,
    )

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    shutil.copy(paths["summary_md"], out_dir / "summary_with_hybrid.md")
    shutil.copy(paths["metrics_json"], out_dir / "metrics_with_hybrid.json")
    shutil.copy(paths["results_csv"], out_dir / "results_with_hybrid.csv")
    shutil.rmtree(tmp_dir)

    print("\n[fcnp] Aggregate results (sorted by F1):")
    print(f"{'Method':<20} {'F1':>8} {'Recall':>8} {'nDCG':>8} {'Comp\u00d7':>8} {'Lat ms':>10}")
    for m in agg:
        print(
            f"{m.method:<20} {m.f1_mean:>8.3f} {m.recall_mean:>8.3f} "
            f"{m.ndcg_mean:>8.3f} {m.compression_ratio_mean:>8.2f} "
            f"{m.latency_ms_mean:>10.2f}"
        )
    print(f"\nwrote results/summary_with_hybrid.md, results/metrics_with_hybrid.json, "
          f"results/results_with_hybrid.csv")


if __name__ == "__main__":
    main()
