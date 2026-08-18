"""Core data types for FCNP."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Tier(str, Enum):
    """Hybrid retention tier assigned to a survivor (improvement #2).

    KEEP_VERBATIM : high flow — full original text preserved.
    SUMMARIZE     : medium flow — compacted text, citation preserved.
    DROP          : near-zero flow — excluded from output entirely.
    PERSISTENT    : promoted into the cross-round memory tier
                    (improvement #3) — always kept verbatim regardless
                    of this round's raw flow.
    """

    KEEP_VERBATIM = "keep_verbatim"
    SUMMARIZE = "summarize"
    DROP = "drop"
    PERSISTENT = "persistent"


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
    tier : Tier | None
        Hybrid retention tier assigned during pruning (None before
        pruning runs). See ``Tier`` and improvement #2.
    summary_text : str | None
        Populated only when ``tier == Tier.SUMMARIZE``; the compacted
        text actually forwarded downstream instead of ``text``.
    """

    id: str
    text: str
    embedding: np.ndarray
    importance: float = 0.5
    citations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    tier: Tier | None = None
    summary_text: str | None = None

    def token_count(self) -> int:
        return max(1, int(len(self.text.split()) * 1.3))

    def output_text(self) -> str:
        """Text actually forwarded downstream, respecting the assigned tier."""
        if self.tier == Tier.SUMMARIZE and self.summary_text is not None:
            return self.summary_text
        return self.text

    def output_token_count(self) -> int:
        return max(1, int(len(self.output_text().split()) * 1.3))


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
    tier_counts: dict[str, int] = field(default_factory=dict)
    persistent_ids: list[str] = field(default_factory=list)
    trigger_reason: str | None = None

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
