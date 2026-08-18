"""Autonomous multi-round context session (ties improvements #1-#3 together).

``FlowBasedNetworkPruner.prune()`` stays a single-shot, stateless call so
existing callers (``fcnp.eval``, ``hf_space/app.py``, the benchmark
baselines) keep working unmodified — it is invoked once per example.

``AutonomousContextSession`` is the new "autonomous memory management"
layer this class of system needs for real multi-turn agent use: it
holds a running context list across turns, feeds the flow-entropy
trigger (improvement #1, ``fcnp.trigger``) every time new elements
arrive, and only re-invokes the pruner when the trigger actually fires
— re-pruning "when it matters", not on a fixed clock. Across
re-prunes, a ``PersistentMemoryTier`` (improvement #3, ``fcnp.memory``)
tracks which elements have earned cross-round persistence, and the
pruner applies hybrid keep/summarize/drop tiering (improvement #2,
``fcnp.pruner`` + ``fcnp.summarize``) within each re-prune.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from fcnp.memory import PersistentMemoryTier
from fcnp.pruner import FCNPConfig, FlowBasedNetworkPruner
from fcnp.trigger import DynamicReprioritizationTrigger, TriggerDecision
from fcnp.types import ContextElement, PruneResult


@dataclass
class AutonomousContextSession:
    """Stateful wrapper for running FCNP across many agent turns.

    Parameters
    ----------
    config : FCNPConfig | None
        Pruner configuration (hybrid tiering fields included). Defaults
        to ``FCNPConfig()`` with hybrid tiering enabled.
    trigger : DynamicReprioritizationTrigger | None
        Defaults to a fresh trigger with library defaults.
    persistent_memory : PersistentMemoryTier | None
        Defaults to a fresh persistent-memory tier with library defaults.
    summarizer : Callable[[str, str | None, int], str] | None
        Forwarded to the pruner for SUMMARIZE-tier text. Defaults to
        ``fcnp.summarize.extractive_summarize`` inside the pruner.
    """

    config: FCNPConfig | None = None
    trigger: DynamicReprioritizationTrigger | None = None
    persistent_memory: PersistentMemoryTier | None = None
    summarizer: Callable[[str, str | None, int], str] | None = None

    _elements: list[ContextElement] = field(default_factory=list, init=False, repr=False)
    _last_result: PruneResult | None = field(default=None, init=False, repr=False)
    _last_node_flow: np.ndarray | None = field(default=None, init=False, repr=False)
    _pruner: FlowBasedNetworkPruner | None = field(default=None, init=False, repr=False)
    round_count: int = field(default=0, init=False)
    reprune_count: int = field(default=0, init=False)

    def __post_init__(self):
        if self.config is None:
            # Hybrid tiering defaults to enabled here (unlike the bare
            # FCNPConfig()/prune() default) because a multi-round
            # autonomous session is exactly the improvement #2 use case:
            # summarize-instead-of-drop for medium-flow context that may
            # matter again later in the same session.
            self.config = FCNPConfig(enable_hybrid_tiering=True)
        if self.trigger is None:
            self.trigger = DynamicReprioritizationTrigger()
        if self.persistent_memory is None:
            self.persistent_memory = PersistentMemoryTier()
        self._pruner = FlowBasedNetworkPruner(
            config=self.config,
            persistent_memory=self.persistent_memory,
            summarizer=self.summarizer,
        )

    def observe(
        self,
        new_elements: list[ContextElement],
        query_embedding: np.ndarray | None = None,
        query_text: str | None = None,
    ) -> PruneResult | None:
        """Append ``new_elements`` to the running context; re-prune only if the trigger fires.

        Returns a fresh ``PruneResult`` when a re-prune actually ran this
        call, or ``None`` when the trigger deferred (no re-prune this
        turn — callers should keep using the previous result via
        ``last_result``).
        """
        self._elements.extend(new_elements)
        self.round_count += 1

        if self._last_node_flow is None:
            # First observation ever: no prior flow distribution to check
            # entropy drift against, so we must prune once to establish
            # a baseline signal for the trigger.
            decision = TriggerDecision(
                should_reprune=True,
                reason="initial context — establishing baseline prune",
                entropy=0.0, entropy_ema=0.0, round_index=self.round_count,
            )
        else:
            # Pad/truncate the last known flow vector to the current
            # element count so entropy is computed over a same-shaped
            # (if approximate) distribution — new elements contribute
            # zero prior flow until the next actual prune assigns them one.
            n = len(self._elements)
            padded = np.zeros(n)
            k = min(n, self._last_node_flow.size)
            padded[:k] = self._last_node_flow[:k]
            decision = self.trigger.observe(padded, n_elements_total=n)

        if not decision.should_reprune:
            return None

        result = self._pruner.prune(
            self._elements, query_embedding=query_embedding, query_text=query_text,
        )
        result.trigger_reason = decision.reason
        self._last_result = result
        self._last_node_flow = result.node_flow
        self.reprune_count += 1

        # Drop DROP-tier elements from the running context so it doesn't
        # grow unboundedly forever; keep everything the pruner actually
        # decided to retain (verbatim, summarized, or persistent).
        self._elements = list(result.survivors)

        return result

    @property
    def last_result(self) -> PruneResult | None:
        return self._last_result

    @property
    def current_context_size(self) -> int:
        return len(self._elements)

    def reset(self) -> None:
        self._elements = []
        self._last_result = None
        self._last_node_flow = None
        self.round_count = 0
        self.reprune_count = 0
        self.trigger.reset()
        self.persistent_memory.reset()
