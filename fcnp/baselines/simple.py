"""Trivial baselines: no compression, random, top-k importance."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

import numpy as np

from fcnp.baselines.base import BaselineResult
from fcnp.types import ContextElement


@dataclass
class NoCompressionBaseline:
    name: str = "NoCompression"

    def compress(self, elements, query_text, query_embedding, keep_k) -> BaselineResult:
        t0 = time.perf_counter()
        survivors = list(elements)
        wall = (time.perf_counter() - t0) * 1000
        in_tok = sum(e.token_count() for e in elements)
        out_tok = sum(e.token_count() for e in survivors)
        return BaselineResult(
            survivors=survivors,
            n_input=len(elements), n_output=len(survivors),
            input_tokens=in_tok, output_tokens=out_tok,
            wall_time_ms=wall,
        )


@dataclass
class RandomBaseline:
    name: str = "Random"
    seed: int = 0

    def compress(self, elements, query_text, query_embedding, keep_k) -> BaselineResult:
        t0 = time.perf_counter()
        rng = random.Random(self.seed)
        k = min(keep_k, len(elements))
        survivors = rng.sample(elements, k) if k > 0 else []
        wall = (time.perf_counter() - t0) * 1000
        in_tok = sum(e.token_count() for e in elements)
        out_tok = sum(e.token_count() for e in survivors)
        return BaselineResult(
            survivors=survivors,
            n_input=len(elements), n_output=len(survivors),
            input_tokens=in_tok, output_tokens=out_tok,
            wall_time_ms=wall,
        )


@dataclass
class TopKImportanceBaseline:
    """Keep the k elements with highest prior importance score."""

    name: str = "TopKImportance"

    def compress(self, elements, query_text, query_embedding, keep_k) -> BaselineResult:
        t0 = time.perf_counter()
        k = min(keep_k, len(elements))
        ranked = sorted(elements, key=lambda e: -e.importance)
        survivors = ranked[:k]
        wall = (time.perf_counter() - t0) * 1000
        in_tok = sum(e.token_count() for e in elements)
        out_tok = sum(e.token_count() for e in survivors)
        return BaselineResult(
            survivors=survivors,
            n_input=len(elements), n_output=len(survivors),
            input_tokens=in_tok, output_tokens=out_tok,
            wall_time_ms=wall,
        )
