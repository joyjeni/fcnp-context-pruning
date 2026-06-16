"""End-to-end synthetic benchmark to verify the publication pipeline.

Mimics the ToolBench evaluation structure with synthetic data that has
clear ground-truth signal, so we can validate the harness end-to-end
without HuggingFace access.
"""

import json
import time
from pathlib import Path

import numpy as np

from fcnp import (
    ALL_BASELINES, FCNPConfig, FCNPMethod,
    ToolBenchExample, evaluate_all, aggregate, write_report,
)


def make_example(seed: int, n_apis: int = 80, n_relevant: int = 5) -> ToolBenchExample:
    rng = np.random.default_rng(seed)
    kw_pool = [
        "weather", "flight", "stock", "translate", "calendar",
        "image", "music", "video", "currency", "news",
        "maps", "search", "email", "sports", "recipe",
    ]
    chosen = rng.choice(len(kw_pool), size=n_relevant, replace=False)
    query_terms = [kw_pool[i] for i in chosen]
    query = "Help me with " + ", ".join(query_terms) + " tasks."

    apis = []
    relevant = []
    # Relevant APIs use real keyword
    for i, kw in enumerate(query_terms):
        api = {
            "category_name": "Productivity",
            "tool_name": f"{kw}_tool",
            "api_name": f"get_{kw}_data",
            "api_description": (
                f"Provides {kw}-related information and operations. "
                f"Use this API to retrieve {kw} records, search {kw} entries, "
                f"and aggregate {kw} statistics."
            ),
            "required_parameters": [{"name": "id", "type": "STRING"}],
        }
        apis.append(api)
        relevant.append((api["tool_name"], api["api_name"]))

    # Distractor APIs
    distractors = [w for w in kw_pool if w not in query_terms]
    for i in range(n_apis - n_relevant):
        dk = distractors[i % len(distractors)]
        api = {
            "category_name": "Misc",
            "tool_name": f"{dk}_distractor_{i}",
            "api_name": f"random_{dk}_{i}",
            "api_description": f"Unrelated {dk} utility for benchmarking.",
            "required_parameters": [],
        }
        apis.append(api)

    rng.shuffle(apis)
    return ToolBenchExample(
        query_id=f"syn-{seed}",
        query=query, api_list=apis,
        relevant_apis=relevant, split="synthetic_g1",
    )


def main():
    out_dir = Path("/home/user/workspace/fcnp")
    print("[fcnp] generating synthetic ToolBench-shaped examples...")
    examples = {"synthetic_g1": [make_example(s) for s in range(30)]}

    fcnp_cfg = FCNPConfig(
        similarity_threshold=0.25,
        max_iterations=120,
        epsilon=1e-3,
        mu=0.10, alpha=0.50, gamma=1.20,
    )
    methods = {name: cls() for name, cls in ALL_BASELINES.items()}
    methods["FCNP"] = FCNPMethod(config=fcnp_cfg)

    print(f"[fcnp] running {len(methods)} methods on {sum(len(v) for v in examples.values())} examples...")
    t0 = time.perf_counter()
    scores = evaluate_all(methods, examples, keep_k_strategy="oracle")
    elapsed = time.perf_counter() - t0
    print(f"[fcnp] {len(scores)} (method, example) evaluations in {elapsed:.1f}s")

    agg = aggregate(scores)
    paths = write_report(
        out_dir, scores, agg,
        primary_method="FCNP",
        dataset_name="ToolBench (synthetic G1)",
        config_used=fcnp_cfg.__dict__,
    )

    print("\n[fcnp] Aggregate results (sorted by F1):")
    print(f"{'Method':<20} {'F1':>8} {'Recall':>8} {'nDCG':>8} {'Comp×':>8} {'Lat ms':>10}")
    for m in agg:
        print(
            f"{m.method:<20} {m.f1_mean:>8.3f} {m.recall_mean:>8.3f} "
            f"{m.ndcg_mean:>8.3f} {m.compression_ratio_mean:>8.2f} "
            f"{m.latency_ms_mean:>10.2f}"
        )

    print("\n[fcnp] Wrote:")
    for k, v in paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
