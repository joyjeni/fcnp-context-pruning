"""Unit tests for the compute-cost vs token-cost model (improvement #5)."""

from fcnp import LLMCompressionCostModel, cost_comparison_table, format_markdown_table


def test_fcnp_has_zero_token_cost():
    comp = cost_comparison_table(
        fcnp_latencies_ms=[5.0, 6.0, 4.5],
        n_reprune_events=3,
    )
    assert comp["fcnp"]["total_token_cost_usd"] == 0.0
    assert comp["llm_based"]["total_token_cost_usd"] > 0.0


def test_latency_speedup_direction():
    comp = cost_comparison_table(
        fcnp_latencies_ms=[2.0, 2.0, 2.0],
        n_reprune_events=3,
        llm_model=LLMCompressionCostModel(avg_llm_latency_ms=900.0),
    )
    # FCNP's linear solve should be dramatically faster than an LLM round trip.
    assert comp["latency_speedup_x"] > 1.0


def test_zero_events_no_crash():
    comp = cost_comparison_table(fcnp_latencies_ms=[], n_reprune_events=0)
    assert comp["n_events"] == 0
    assert comp["fcnp"]["total_latency_ms"] == 0.0


def test_markdown_table_has_expected_rows():
    comp = cost_comparison_table(fcnp_latencies_ms=[3.0], n_reprune_events=1)
    table = format_markdown_table(comp)
    assert "Total latency" in table
    assert "Token cost" in table
    assert "|" in table
