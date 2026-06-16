"""Core data types for FCNP."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ContextElement:
    """A single unit of information in the context store.

    Attributes
    ----------
    id : str
        Unique identifier.
    text : str
        Raw textual content.
    embedding : np.ndarray
        Dense semantic vector used to compute edge weights.
    importance : float
        Prior salience in [0, 1]; updated to reflect computed node flow
        after pruning.
    citations : list[str]
        Source identifiers preserved through compression.
    metadata : dict
        Free-form annotations.
    """

    id: str
    text: str
    embedding: np.ndarray
    importance: float = 0.5
    citations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def token_count(self) -> int:
        return max(1, int(len(self.text.split()) * 1.3))


@dataclass
class PruneResult:
    """Output of FlowBasedNetworkPruner.prune()."""

    survivors: list[ContextElement]
    n_input: int
    n_output: int
    input_tokens: int
    output_tokens: int
    iterations: int
    converged: bool
    node_flow: np.ndarray  # aggregate flow per input element

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
