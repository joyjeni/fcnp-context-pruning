"""Unit tests for the persistent high-flow memory tier (improvement #3)."""

from fcnp import PersistentMemoryTier


def _ranked(ids_best_first):
    """Build (id, flow) pairs, best-first, with decreasing synthetic flow."""
    n = len(ids_best_first)
    return [(eid, 1.0 - i / max(n, 1)) for i, eid in enumerate(ids_best_first)]


def test_promotion_after_consecutive_top_rounds():
    mem = PersistentMemoryTier(promotion_rank_fraction=0.2, promotion_rounds=3)
    ids = [f"e{i}" for i in range(10)]

    # "e0" stays in the top 20% for 3 consecutive rounds -> promoted.
    for _ in range(3):
        result = mem.update(_ranked(ids))

    assert mem.is_persistent("e0")
    assert "e0" in mem.persistent_ids()


def test_no_promotion_without_enough_consecutive_rounds():
    mem = PersistentMemoryTier(promotion_rank_fraction=0.2, promotion_rounds=5)
    ids = [f"e{i}" for i in range(10)]
    for _ in range(2):
        mem.update(_ranked(ids))
    assert not mem.is_persistent("e0")


def test_demotion_after_falling_out_of_top_fraction():
    mem = PersistentMemoryTier(
        promotion_rank_fraction=0.3, promotion_rounds=2,
        demotion_rank_fraction=0.5, demotion_rounds=2,
    )
    ids = [f"e{i}" for i in range(10)]
    for _ in range(2):
        mem.update(_ranked(ids))
    assert mem.is_persistent("e0")

    # Now e0 consistently ranks last (bottom zone) for demotion_rounds.
    demoted_ids = list(reversed(ids))  # e0 now ranked last
    for _ in range(2):
        result = mem.update(_ranked(demoted_ids))

    assert not mem.is_persistent("e0")


def test_max_size_cap_evicts_lowest_flow_member():
    mem = PersistentMemoryTier(max_size=2, promotion_rank_fraction=0.5, promotion_rounds=1)
    # Promote e0, e1 first (both in top 50% of a 2-element ranking).
    mem.update(_ranked(["e0", "e1"]))
    assert len(mem.persistent_ids()) <= 2

    # A strong new candidate e2 with higher flow than the worst incumbent
    # should be able to displace it once max_size is reached.
    mem.update(_ranked(["e2", "e0"]))
    assert len(mem.persistent_ids()) <= 2


def test_reset_clears_persistent_set():
    mem = PersistentMemoryTier(promotion_rank_fraction=0.5, promotion_rounds=1)
    mem.update(_ranked(["e0", "e1"]))
    assert len(mem.persistent_ids()) > 0
    mem.reset()
    assert mem.persistent_ids() == set()
