"""Unit tests for FCNP pruner."""

import numpy as np
import pytest

from fcnp import (
    ContextElement,
    FCNPConfig,
    FlowBasedNetworkPruner,
)


def _make_elements(n=50, dim=32, seed=0):
    rng = np.random.default_rng(seed)
    return [
        ContextElement(
            id=f"e{i}",
            text=f"element {i} content with some words for token count",
            embedding=rng.standard_normal(dim),
            importance=0.5,
            citations=[f"src{i}"],
        )
        for i in range(n)
    ]


def test_empty():
    p = FlowBasedNetworkPruner()
    r = p.prune([])
    assert r.n_input == 0 and r.n_output == 0
    assert r.converged


def test_converges():
    elements = _make_elements(40)
    p = FlowBasedNetworkPruner(FCNPConfig(max_iterations=300, epsilon=1e-3))
    r = p.prune(elements)
    assert r.iterations <= 300
    assert r.n_output <= r.n_input


def test_keep_top_k_fraction():
    elements = _make_elements(100)
    p = FlowBasedNetworkPruner(FCNPConfig(keep_top_k_fraction=0.1))
    r = p.prune(elements)
    assert r.n_output == 10


def test_compression_ratio_positive():
    elements = _make_elements(50)
    p = FlowBasedNetworkPruner(FCNPConfig(keep_top_k_fraction=0.1))
    r = p.prune(elements)
    assert r.compression_ratio > 1.0
    assert 0 <= r.reduction_pct <= 100


def test_node_flow_shape():
    elements = _make_elements(30)
    p = FlowBasedNetworkPruner(FCNPConfig(keep_top_k_fraction=0.2))
    r = p.prune(elements)
    assert r.node_flow.shape == (30,)
    assert (r.node_flow >= 0).all()


def test_query_steers_selection():
    """An element semantically aligned with the query should rank highly."""
    rng = np.random.default_rng(42)
    base = [ContextElement(id=f"e{i}", text=f"t{i}", embedding=rng.standard_normal(16))
            for i in range(40)]
    target_vec = rng.standard_normal(16)
    base[5].embedding = target_vec * 5  # plant a strong source
    base[5].text = "the target element with extra tokens for weight"

    p = FlowBasedNetworkPruner(FCNPConfig(keep_top_k_fraction=0.2))
    r = p.prune(base, query_embedding=target_vec)
    kept_ids = {e.id for e in r.survivors}
    assert "e5" in kept_ids


def test_citations_preserved_in_survivors():
    elements = _make_elements(40)
    # Mark first 4 as "critical" with unique citations
    for i in range(4):
        elements[i].importance = 1.0
        elements[i].citations = [f"crit{i}"]
        elements[i].embedding = np.ones(elements[i].embedding.shape) * (i + 1)
    p = FlowBasedNetworkPruner(FCNPConfig(keep_top_k_fraction=0.2))
    r = p.prune(elements, query_embedding=np.ones(elements[0].embedding.shape))
    kept_cites = {c for e in r.survivors for c in e.citations}
    # At least one critical citation must survive
    assert kept_cites & {f"crit{i}" for i in range(4)}
