"""Unit tests for AutonomousContextSession (multi-round orchestration
tying together improvements #1-#3)."""

import numpy as np

from fcnp import AutonomousContextSession, ContextElement, FCNPConfig


def _mk(eid, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return ContextElement(id=eid, text=f"content for {eid} " * 5, embedding=rng.standard_normal(dim))


def test_first_observation_always_prunes():
    session = AutonomousContextSession(config=FCNPConfig(max_iterations=30))
    batch = [_mk(f"e{i}", seed=i) for i in range(10)]
    result = session.observe(batch)
    assert result is not None
    assert session.reprune_count == 1
    assert session.round_count == 1


def test_context_shrinks_to_survivors_after_reprune():
    session = AutonomousContextSession(config=FCNPConfig(max_iterations=30, keep_top_k_fraction=0.1))
    batch = [_mk(f"e{i}", seed=i) for i in range(30)]
    result = session.observe(batch)
    assert result is not None
    assert session.current_context_size == result.n_output
    assert session.current_context_size < 30


def test_reset_clears_all_state():
    session = AutonomousContextSession()
    batch = [_mk(f"e{i}", seed=i) for i in range(5)]
    session.observe(batch)
    assert session.round_count == 1
    session.reset()
    assert session.round_count == 0
    assert session.reprune_count == 0
    assert session.current_context_size == 0
    assert session.last_result is None


def test_deferred_round_returns_none_and_keeps_growing_context():
    # Use a trigger that essentially never fires on drift so we can
    # observe a deferred (None) round distinct from the mandatory first prune.
    from fcnp import DynamicReprioritizationTrigger

    trig = DynamicReprioritizationTrigger(
        drift_threshold=100.0, stagnation_rounds=100, min_growth_elements=10_000,
    )
    session = AutonomousContextSession(
        config=FCNPConfig(max_iterations=30), trigger=trig,
    )
    first = session.observe([_mk(f"a{i}", seed=i) for i in range(10)])
    assert first is not None  # baseline prune always happens

    second = session.observe([_mk(f"b{i}", seed=100 + i) for i in range(3)])
    assert second is None  # trigger deferred
    assert session.current_context_size > first.n_output  # context kept growing
