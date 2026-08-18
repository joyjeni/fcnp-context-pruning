"""FCNP with hybrid keep/summarize/drop tiering (improvement #2), exposed
under the Baseline interface as its own benchmark row.

Kept separate from ``FCNPMethod`` (fcnp_wrapper.py) so the original
strict top-K "FCNP" benchmark numbers stay exactly reproducible while
this row demonstrates the tradeoff hybrid tiering buys: it trades a
larger output-token budget (verbatim + summarized survivors, instead
of a hard top-K cutoff) for higher recall — directly addressing the
context-loss failure mode Focus (arXiv:2601.07190) reports in its own
ablation (the ``pylint-7080`` case: a dropped item forced +110% tokens
of re-exploration later). ``recall``/``precision`` here are computed
over the *union* of KEEP_VERBATIM + SUMMARIZE + PERSISTENT survivors,
i.e. "was the relevant item retained in some form", while
``output_tokens`` reflects the actually-cheaper summarized text length
for medium-flow items rather than their full original length.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from fcnp.baselines.base import BaselineResult
from fcnp.pruner import FCNPConfig, FlowBasedNetworkPruner


@dataclass
class FCNPHybridMethod:
    name: str = "FCNP-Hybrid"
    config: FCNPConfig = None
    summarize_top_k_fraction: float = 0.20

    def __post_init__(self):
        if self.config is None:
            self.config = FCNPConfig()

    def compress(self, elements, query_text, query_embedding, keep_k) -> BaselineResult:
        cfg = FCNPConfig(
            similarity_threshold=self.config.similarity_threshold,
            epsilon=self.config.epsilon,
            max_iterations=self.config.max_iterations,
            mu=self.config.mu,
            alpha=self.config.alpha,
            gamma=self.config.gamma,
            keep_top_k_fraction=keep_k / max(len(elements), 1),
            summarize_top_k_fraction=self.summarize_top_k_fraction,
            current_injection=self.config.current_injection,
            laplacian_regularization=self.config.laplacian_regularization,
            enable_hybrid_tiering=True,
        )
        pruner = FlowBasedNetworkPruner(cfg)
        t0 = time.perf_counter()
        result = pruner.prune(elements, query_embedding=query_embedding, query_text=query_text)
        wall = (time.perf_counter() - t0) * 1000
        return BaselineResult(
            survivors=result.survivors,
            n_input=result.n_input,
            n_output=result.n_output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            wall_time_ms=wall,
        )
