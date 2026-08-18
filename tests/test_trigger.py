"""Unit tests for the dynamic flow-entropy re-pruning trigger (improvement #1)."""

import numpy as np

from fcnp import DynamicReprioritizationTrigger, flow_entropy


def test_flow_entropy_bounds():
    n = 20
    concentrated = np.zeros(n)
    concentrated[0] = 1.0
    assert flow_entropy(concentrated) == 0.0

    uniform = np.ones(n)
    assert flow_entropy(uniform) > 0.99

    assert flow_entropy(np.zeros(n)) == 0.0
    assert flow_entropy(np.zeros(1)) == 0.0


def test_first_observation_never_fires():
    trig = DynamicReprioritizationTrigger()
    decision = trig.observe(np.ones(10), n_elements_total=10)
    assert decision.should_reprune is False
    assert "baseline" in decision.reason


def test_drift_trigger_fires_on_entropy_jump():
    trig = DynamicReprioritizationTrigger(drift_threshold=0.10, ema_alpha=0.5)
    n = 20
    concentrated = np.zeros(n)
    concentrated[0] = 1.0
    trig.observe(concentrated, n_elements_total=n)  # baseline

    uniform = np.ones(n)
    decision = trig.observe(uniform, n_elements_total=n)
    assert decision.should_reprune is True
    assert "diffusing" in decision.reason or "drift" in decision.reason


def test_stagnation_backstop_fires_after_flat_rounds_and_growth():
    trig = DynamicReprioritizationTrigger(
        drift_threshold=10.0,  # effectively disable the drift trigger
        stagnation_rounds=2,
        stagnation_flat_tol=0.05,
        min_growth_elements=3,
    )
    n = 10
    flow = np.ones(n)
    trig.observe(flow, n_elements_total=n)  # baseline

    # Flat entropy, but not enough growth yet.
    d1 = trig.observe(np.ones(n + 1), n_elements_total=n + 1)
    assert d1.should_reprune is False

    # Flat entropy and growth beyond min_growth_elements over stagnation_rounds.
    d2 = trig.observe(np.ones(n + 5), n_elements_total=n + 5)
    assert d2.should_reprune is True
    assert "stagnation" in d2.reason.lower() or "backstop" in d2.reason.lower()


def test_reset_clears_state():
    trig = DynamicReprioritizationTrigger()
    trig.observe(np.ones(5), n_elements_total=5)
    trig.observe(np.ones(6), n_elements_total=6)
    trig.reset()
    decision = trig.observe(np.ones(5), n_elements_total=5)
    assert "baseline" in decision.reason
