"""Compute-cost vs token-cost quantification (improvement #5).

FCNP's cost is a deterministic linear solve — no model calls, so its
cost is pure wall-clock compute (already tracked as ``latency_ms`` in
``fcnp.eval``). LLM-based compressors such as Focus
(arXiv:2601.07190, "Active Context Compression: Autonomous Memory
Management in LLM Agents") instead spend *tokens*: every compression
event is itself an LLM call that reads the current context and writes
a compacted replacement, so its cost is latency **and** a metered
token bill that scales with provider pricing and context size.

This module makes that difference explicit and auditable rather than
asserting "FCNP is cheaper" as a slogan. Every dollar/latency number
below is a labeled input you can override — nothing here claims to be
a measurement of Focus's actual production costs, which were not
published in the paper as a $/session figure.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMCompressionCostModel:
    """Illustrative cost model for an LLM-driven compression call.

    Defaults are labeled placeholders (roughly mid-2026 mid-tier hosted
    LLM pricing order-of-magnitude) — override with your own provider's
    published rates for a real comparison; do not cite the defaults as
    fact.
    """

    input_cost_per_1k_tokens_usd: float = 0.003
    output_cost_per_1k_tokens_usd: float = 0.015
    avg_context_tokens_read: int = 2000   # tokens the LLM must read to compress
    avg_summary_tokens_written: int = 200  # tokens it writes back
    avg_llm_latency_ms: float = 900.0      # network + generation latency, one call

    def cost_per_call_usd(self) -> float:
        read_cost = (self.avg_context_tokens_read / 1000.0) * self.input_cost_per_1k_tokens_usd
        write_cost = (self.avg_summary_tokens_written / 1000.0) * self.output_cost_per_1k_tokens_usd
        return read_cost + write_cost


@dataclass
class FCNPCostModel:
    """FCNP's own cost is just its measured linear-solve latency.

    No token cost exists because no model is called during pruning —
    only a dense embedding lookup (already-computed / cached) and one
    Laplacian solve per round.
    """

    measured_latency_ms: float
    embedding_amortized: bool = True  # embeddings usually already cached from retrieval


def cost_comparison_table(
    fcnp_latencies_ms: list[float],
    n_reprune_events: int,
    llm_model: LLMCompressionCostModel | None = None,
) -> dict:
    """Build a side-by-side cost table for N re-compression events.

    Parameters
    ----------
    fcnp_latencies_ms : per-event measured FCNP solve latency (ms).
    n_reprune_events : number of times compression fires over a session
        — pass FCNP's own dynamic-trigger fire count for a fair
        apples-to-apples comparison at equal trigger frequency.
    llm_model : override the default illustrative LLM cost model.
    """
    llm_model = llm_model or LLMCompressionCostModel()

    fcnp_total_latency_ms = sum(fcnp_latencies_ms[:n_reprune_events]) if fcnp_latencies_ms else 0.0
    fcnp_avg_latency_ms = (
        fcnp_total_latency_ms / max(1, min(n_reprune_events, len(fcnp_latencies_ms)))
        if fcnp_latencies_ms
        else 0.0
    )

    llm_cost_per_call = llm_model.cost_per_call_usd()
    llm_total_cost_usd = llm_cost_per_call * n_reprune_events
    llm_total_latency_ms = llm_model.avg_llm_latency_ms * n_reprune_events

    return {
        "n_events": n_reprune_events,
        "fcnp": {
            "total_latency_ms": fcnp_total_latency_ms,
            "avg_latency_ms": fcnp_avg_latency_ms,
            "total_token_cost_usd": 0.0,
            "notes": "deterministic linear solve; no LLM call; no token bill",
        },
        "llm_based": {
            "total_latency_ms": llm_total_latency_ms,
            "avg_latency_ms": llm_model.avg_llm_latency_ms,
            "total_token_cost_usd": llm_total_cost_usd,
            "cost_per_event_usd": llm_cost_per_call,
            "notes": (
                f"illustrative: reads ~{llm_model.avg_context_tokens_read} tok, "
                f"writes ~{llm_model.avg_summary_tokens_written} tok per compression call"
            ),
        },
        "latency_speedup_x": (
            llm_total_latency_ms / fcnp_total_latency_ms if fcnp_total_latency_ms > 0 else float("inf")
        ),
    }


def format_markdown_table(comparison: dict) -> str:
    f = comparison["fcnp"]
    l = comparison["llm_based"]
    n = comparison["n_events"]
    speedup = comparison["latency_speedup_x"]
    lines = [
        f"| Metric | FCNP ({n} events) | LLM-based compression ({n} events) |",
        "|---|---|---|",
        f"| Total latency | {f['total_latency_ms']:.1f} ms | {l['total_latency_ms']:.1f} ms |",
        f"| Avg latency / event | {f['avg_latency_ms']:.2f} ms | {l['avg_latency_ms']:.1f} ms |",
        f"| Token cost | $0.00 (no LLM call) | ${l['total_token_cost_usd']:.4f} (${l['cost_per_event_usd']:.5f}/event) |",
        f"| Mechanism | Laplacian linear solve | LLM read+summarize call |",
    ]
    if speedup != float("inf"):
        lines.append(f"| Latency speedup | **{speedup:.0f}x** faster | baseline |")
    return "\n".join(lines)
