"""Smoke tests for all baselines + FCNP wrapper."""

import numpy as np
import pytest

from fcnp.baselines import ALL_BASELINES
from fcnp.baselines.fcnp_wrapper import FCNPMethod
from fcnp.datasets.toolbench import _hash_embedding
from fcnp.types import ContextElement


def _elements(n=30):
    return [
        ContextElement(
            id=f"e{i}",
            text=f"this is element {i} with some keywords like alpha beta gamma",
            embedding=_hash_embedding(f"e{i}", dim=32),
            importance=float(i) / n,
            citations=[f"e{i}"],
        )
        for i in range(n)
    ]


@pytest.mark.parametrize("name", list(ALL_BASELINES.keys()))
def test_baseline_runs(name):
    method = ALL_BASELINES[name]()
    elements = _elements(40)
    q_emb = _hash_embedding("alpha beta", dim=32)
    r = method.compress(elements, "alpha beta", q_emb, keep_k=10)
    if name == "NoCompression":
        assert r.n_output == 40
    else:
        assert r.n_output == 10


def test_fcnp_method_runs():
    m = FCNPMethod()
    elements = _elements(30)
    q_emb = _hash_embedding("alpha beta", dim=32)
    r = m.compress(elements, "alpha beta", q_emb, keep_k=5)
    assert r.n_output == 5
    assert r.compression_ratio > 1.0
