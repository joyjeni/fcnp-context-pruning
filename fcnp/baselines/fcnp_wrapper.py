"""FCNP exposed under the Baseline interface for uniform benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from fcnp.baselines.base import BaselineResult
from fcnp.pruner import FCNPConfig, FlowBasedNetworkPruner
from fcnp.types import ContextElement


@dataclass
class FCNPMethod:
    name: str = "FCNP"
    config: FCNPConfig = None

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
            current_injection=self.config.current_injection,
            laplacian_regularization=self.config.laplacian_regularization,
        )
        pruner = FlowBasedNetworkPruner(cfg)
        t0 = time.perf_counter()
        result = pruner.prune(elements, query_embedding=query_embedding)
        wall = (time.perf_counter() - t0) * 1000
        return BaselineResult(
            survivors=result.survivors,
            n_input=result.n_input,
            n_output=result.n_output,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            wall_time_ms=wall,
        )
