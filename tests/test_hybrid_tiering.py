"""Unit tests for hybrid keep/summarize/drop tiering (improvement #2) and
the persistent-memory integration (improvement #3) inside prune()."""

import numpy as np

from fcnp import ContextElement, FCNPConfig, FlowBasedNetworkPruner, PersistentMemoryTier
from fcnp.types import Tier


def _make_elements(n=50, dim=32, seed=0, word_len=40):
    rng = np.random.default_rng(seed)
    return [
        ContextElement(
            id=f"e{i}",
            text=" ".join([f"word{j} number {i} filler." for j in range(word_len)]),
            embedding=rng.standard_normal(dim),
            importance=0.5,
        )
        for i in range(n)
    ]


def test_hybrid_tiering_disabled_by_default():
    elements = _make_elements(100)
    p = FlowBasedNetworkPruner(FCNPConfig(keep_top_k_fraction=0.1))
    r = p.prune(elements)
    assert r.n_output == 10
    assert r.tier_counts.get("summarize", 0) == 0


def test_hybrid_tiering_adds_summarize_tier_when_enabled():
    elements = _make_elements(100)
    cfg = FCNPConfig(keep_top_k_fraction=0.1, summarize_top_k_fraction=0.2, enable_hybrid_tiering=True)
    p = FlowBasedNetworkPruner(cfg)
    r = p.prune(elements, query_text="word1")
    assert r.tier_counts.get("keep_verbatim", 0) == 10
    assert r.tier_counts.get("summarize", 0) == 20
    assert r.n_output == 30
    summarized = [e for e in r.survivors if e.tier == Tier.SUMMARIZE]
    assert all(e.summary_text is not None for e in summarized)
    # Output tokens should reflect summarized (shorter) text for those elements.
    assert r.output_tokens < sum(e.token_count() for e in r.survivors)


def test_persistent_memory_force_includes_low_rank_element():
    elements = _make_elements(20, word_len=10)
    mem = PersistentMemoryTier(promotion_rank_fraction=0.1, promotion_rounds=1)
    cfg = FCNPConfig(keep_top_k_fraction=0.1, enable_hybrid_tiering=True)
    p = FlowBasedNetworkPruner(cfg, persistent_memory=mem)

    r1 = p.prune(elements)
    assert len(r1.persistent_ids) >= 1
    persistent_id = r1.persistent_ids[0]

    # Force that element to look irrelevant next round by giving every
    # other element an overwhelming embedding push, without changing
    # the persistent element itself.
    rng = np.random.default_rng(99)
    for e in elements:
        if e.id != persistent_id:
            e.embedding = e.embedding + rng.standard_normal(e.embedding.shape) * 5

    r2 = p.prune(elements)
    survivor_ids = [e.id for e in r2.survivors]
    assert persistent_id in survivor_ids
    tier_of_persistent = next(e.tier for e in r2.survivors if e.id == persistent_id)
    assert tier_of_persistent == Tier.PERSISTENT
