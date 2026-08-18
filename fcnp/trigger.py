"""Dynamic flow-convergence re-pruning trigger.

Improvement #1 over fixed-interval / fixed-turn-count re-compression
(as used e.g. in "Active Context Compression: Autonomous Memory
Management in LLM Agents", arXiv:2601.07190 — the Focus system there
re-compresses on a static schedule and explicitly reports that *when*
compression fires matters more than *whether* it fires at all).

FCNP already produces, for free, a per-round node-flow distribution
from the resistor-network solve. This module turns that distribution
into a *signal* rather than a byproduct: we track the Shannon entropy
of the normalized flow distribution across rounds and fire a re-prune
only when that signal indicates the context has genuinely drifted or
concentrated — not on a fixed clock.

Two independent triggering conditions (either is sufficient):

1. **Drift trigger** — normalized flow entropy this round differs from
   the trailing EMA of entropy by more than ``drift_threshold``. A large
   entropy *increase* means flow has spread out (many elements now look
   equally relevant — likely a topic shift, worth re-pruning to find the
   new signal). A large entropy *decrease* means flow has sharply
   concentrated (a few elements have become dominant — safe/cheap to
   compact hard around them).
2. **Stagnation trigger** — entropy has stayed essentially flat for
   ``stagnation_rounds`` consecutive rounds while the raw element count
   keeps growing past ``min_growth_elements``; context is quietly
   bloating without the flow signal reorganizing, so a scheduled
   catch-up prune is still useful as a backstop.

This keeps the "whether" cheap (no LLM call, just an entropy computation
already implied by data FCNP has in hand) while making "when" adaptive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def flow_entropy(node_flow: np.ndarray) -> float:
    """Normalized Shannon entropy (in [0, 1]) of the node-flow distribution.

    0.0  -> all flow concentrated on a single element (maximally decisive)
    1.0  -> flow spread uniformly across every element (maximally diffuse)
    """
    flow = np.asarray(node_flow, dtype=np.float64)
    flow = np.clip(flow, 0.0, None)
    total = flow.sum()
    n = flow.size
    if n <= 1 or total <= 0:
        return 0.0
    p = flow / total
    p = p[p > 0]
    h = float(-(p * np.log(p)).sum())
    h_max = float(np.log(n))
    return h / h_max if h_max > 0 else 0.0


@dataclass
class TriggerDecision:
    should_reprune: bool
    reason: str
    entropy: float
    entropy_ema: float
    round_index: int


@dataclass
class DynamicReprioritizationTrigger:
    """Stateful, per-session trigger deciding when FCNP should re-run.

    Parameters
    ----------
    drift_threshold : float
        Minimum |entropy - ema| to fire the drift trigger.
    ema_alpha : float
        Exponential moving average smoothing factor for entropy.
    stagnation_rounds : int
        Consecutive rounds of near-flat entropy before the stagnation
        backstop is armed.
    stagnation_flat_tol : float
        Max entropy delta between consecutive rounds still considered
        "flat" for stagnation counting.
    min_growth_elements : int
        Minimum growth in raw context-element count (since the last
        prune) required for the stagnation trigger to fire — avoids
        re-pruning a session that simply isn't accumulating new context.
    """

    drift_threshold: float = 0.12
    ema_alpha: float = 0.30
    stagnation_rounds: int = 3
    stagnation_flat_tol: float = 0.02
    min_growth_elements: int = 5

    _ema: float | None = field(default=None, init=False, repr=False)
    _flat_streak: int = field(default=0, init=False, repr=False)
    _round: int = field(default=0, init=False, repr=False)
    _last_entropy: float | None = field(default=None, init=False, repr=False)
    _elements_at_last_prune: int = field(default=0, init=False, repr=False)

    def observe(self, node_flow: np.ndarray, n_elements_total: int) -> TriggerDecision:
        """Feed a new round's node-flow distribution; get a fire/no-fire decision.

        Call this every time new context elements are appended, even if
        you don't ultimately re-prune — the entropy trace is what makes
        the *next* decision meaningful.
        """
        entropy = flow_entropy(node_flow)
        self._round += 1

        if self._ema is None:
            self._ema = entropy
            self._last_entropy = entropy
            self._elements_at_last_prune = n_elements_total
            return TriggerDecision(
                should_reprune=False,
                reason="first observation — establishing entropy baseline",
                entropy=entropy,
                entropy_ema=self._ema,
                round_index=self._round,
            )

        drift = abs(entropy - self._ema)
        flat = abs(entropy - (self._last_entropy or entropy)) <= self.stagnation_flat_tol
        self._flat_streak = self._flat_streak + 1 if flat else 0

        growth = n_elements_total - self._elements_at_last_prune
        drift_fired = drift >= self.drift_threshold
        stagnation_fired = (
            self._flat_streak >= self.stagnation_rounds
            and growth >= self.min_growth_elements
        )

        should = drift_fired or stagnation_fired
        if drift_fired and entropy > self._ema:
            reason = (
                f"entropy drift +{drift:.3f} (>= {self.drift_threshold}) — "
                "flow diffusing, likely topic shift: re-prune to re-find signal"
            )
        elif drift_fired:
            reason = (
                f"entropy drift -{drift:.3f} (>= {self.drift_threshold}) — "
                "flow concentrating: safe to compact hard around current winners"
            )
        elif stagnation_fired:
            reason = (
                f"entropy flat for {self._flat_streak} rounds while context grew "
                f"by {growth} elements — scheduled backstop prune"
            )
        else:
            reason = "no drift or stagnation signal — deferring re-prune"

        self._ema = (1 - self.ema_alpha) * self._ema + self.ema_alpha * entropy
        self._last_entropy = entropy
        if should:
            self._elements_at_last_prune = n_elements_total
            self._flat_streak = 0

        return TriggerDecision(
            should_reprune=should,
            reason=reason,
            entropy=entropy,
            entropy_ema=self._ema,
            round_index=self._round,
        )

    def reset(self) -> None:
        self._ema = None
        self._flat_streak = 0
        self._round = 0
        self._last_entropy = None
        self._elements_at_last_prune = 0
