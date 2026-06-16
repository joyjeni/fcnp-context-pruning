"""End-to-end evaluation pipeline test using a synthetic ToolBench-shaped fixture."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fcnp import (
    FCNPConfig, FCNPMethod, ALL_BASELINES,
    ToolBenchExample, evaluate_all, aggregate, write_report,
)


def _synthetic_example(query_id: str, n_apis: int = 50, n_relevant: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    apis = []
    relevant = []
    keywords = ["alpha", "beta", "gamma", "delta", "epsilon"]
    chosen = rng.choice(len(keywords), size=n_relevant, replace=False)
    query_terms = [keywords[i] for i in chosen]
    query = f"find APIs that handle {' '.join(query_terms)}"
    for i in range(n_apis):
        is_rel = i < n_relevant
        kw = query_terms[i] if is_rel else f"unrelated_{rng.integers(1000)}"
        api = {
            "category_name": "TestCat",
            "tool_name": f"tool_{i}",
            "api_name": f"api_{i}",
            "api_description": f"This API handles {kw} operations and related work.",
            "required_parameters": [],
        }
        apis.append(api)
        if is_rel:
            relevant.append((api["tool_name"], api["api_name"]))
    return ToolBenchExample(
        query_id=query_id, query=query,
        api_list=apis, relevant_apis=relevant, split="synthetic",
    )


def test_evaluate_all_and_report():
    examples = {"synthetic": [_synthetic_example(f"q{i}", seed=i) for i in range(8)]}
    methods = {name: cls() for name, cls in ALL_BASELINES.items()}
    methods["FCNP"] = FCNPMethod(config=FCNPConfig(max_iterations=80))

    scores = evaluate_all(methods, examples, keep_k_strategy="oracle")
    assert scores
    agg = aggregate(scores)
    assert {m.method for m in agg} == set(methods.keys())

    with tempfile.TemporaryDirectory() as td:
        paths = write_report(td, scores, agg, primary_method="FCNP", dataset_name="synthetic")
        assert paths["metrics_json"].exists()
        payload = json.loads(paths["metrics_json"].read_text())
        assert payload["primary_method"] == "FCNP"
        assert len(payload["methods"]) == len(methods)
        assert paths["results_csv"].exists()
        assert paths["summary_md"].exists()
