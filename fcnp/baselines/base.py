"""Baseline interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from fcnp.types import ContextElement


@dataclass
class BaselineResult:
    survivors: list[ContextElement]
    n_input: int
    n_output: int
    input_tokens: int
    output_tokens: int
    wall_time_ms: float

    @property
    def compression_ratio(self) -> float:
        if self.output_tokens == 0:
            return float("inf")
        return self.input_tokens / self.output_tokens

    @property
    def reduction_pct(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return 100.0 * (1.0 - self.output_tokens / self.input_tokens)


class Baseline(Protocol):
    """Compression baseline interface."""

    name: str

    def compress(
        self,
        elements: list[ContextElement],
        query_text: str,
        query_embedding: np.ndarray | None,
        keep_k: int,
    ) -> BaselineResult: ...
