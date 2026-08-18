"""Persistent high-flow memory tier.

Improvement #3 — a principled, flow-driven analog of the free-text
"Knowledge" block used in Focus (arXiv:2601.07190) for facts the agent
should never lose. Focus's Knowledge block is manually curated
free-text with no eviction mechanism; here, promotion and eviction are
both *earned* by the same current-reinforcement signal FCNP already
computes, so the persistent set stays small, adapts as the session
shifts, and never silently grows unbounded.

Mechanics
---------
- Every prune round, each surviving element's normalized node flow is
  recorded into a short rolling history per element id.
- An element is **promoted** into the persistent tier once it has
  scored in the top ``promotion_rank_fraction`` of flow for
  ``promotion_rounds`` consecutive rounds it appeared in.
- Promoted elements are exempt from the normal top-K cutoff: they are
  always re-injected as strong current sources on subsequent rounds
  (so they keep influencing the graph even if a single round's raw
  similarity would have dropped them), and their text is always kept
  verbatim (see ``fcnp.pruner`` hybrid tiering).
- An element is **demoted** (evicted from the persistent tier) if its
  flow rank falls out of the top ``demotion_rank_fraction`` for
  ``demotion_rounds`` consecutive rounds — so stale "knowledge" that
  stopped mattering is released instead of accumulating forever.
- The persistent tier has a hard ``max_size`` cap; if promotion would
  exceed it, the lowest-current-flow persistent member is evicted to
  make room (LRU-by-flow, not LRU-by-time).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class _ElementHistory:
    recent_ranks: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    consecutive_top: int = 0
    consecutive_bottom: int = 0


@dataclass
class PersistentMemoryTier:
    max_size: int = 12
    promotion_rank_fraction: float = 0.20   # top 20% of flow this round
    promotion_rounds: int = 3
    demotion_rank_fraction: float = 0.60    # falls out of top 60%
    demotion_rounds: int = 4

    _history: dict[str, _ElementHistory] = field(default_factory=dict, init=False, repr=False)
    _persistent: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    # id -> last-known normalized flow score, for eviction-by-lowest-flow

    def update(self, ranked_ids_with_flow: list[tuple[str, float]]) -> dict[str, list[str]]:
        """Feed one round's (element_id, normalized_flow) pairs, best-first.

        Returns a dict with ``"promoted"`` and ``"demoted"`` id lists for
        this round (for logging / dashboards).
        """
        n = len(ranked_ids_with_flow)
        promoted: list[str] = []
        demoted: list[str] = []
        if n == 0:
            return {"promoted": promoted, "demoted": demoted}

        top_cut = max(1, int(n * self.promotion_rank_fraction))
        bottom_cut = max(1, int(n * self.demotion_rank_fraction))

        seen_ids: set[str] = set()
        for rank, (eid, flow) in enumerate(ranked_ids_with_flow):
            seen_ids.add(eid)
            hist = self._history.setdefault(eid, _ElementHistory())
            hist.recent_ranks.append(rank / max(n - 1, 1))

            in_top = rank < top_cut
            in_bottom_zone = rank >= bottom_cut

            hist.consecutive_top = hist.consecutive_top + 1 if in_top else 0
            hist.consecutive_bottom = hist.consecutive_bottom + 1 if in_bottom_zone else 0

            if eid in self._persistent:
                self._persistent[eid] = flow
                if hist.consecutive_bottom >= self.demotion_rounds:
                    del self._persistent[eid]
                    hist.consecutive_bottom = 0
                    demoted.append(eid)
                continue

            if hist.consecutive_top >= self.promotion_rounds:
                self._promote(eid, flow)
                promoted.append(eid)
                hist.consecutive_top = 0

        # Elements that vanished from context entirely (didn't appear this
        # round) but were persistent: keep them for now — they're exempt
        # from *this round's* graph, but if they never come back they'll
        # simply age out of relevance naturally; we don't force-decay here
        # to keep the mechanism strictly flow-earned, not time-earned.
        return {"promoted": promoted, "demoted": demoted}

    def _promote(self, eid: str, flow: float) -> None:
        if len(self._persistent) >= self.max_size:
            # Evict the current lowest-flow persistent member to make room.
            worst_id = min(self._persistent, key=lambda k: self._persistent[k])
            if self._persistent[worst_id] >= flow:
                return  # new candidate isn't even better than the worst incumbent
            del self._persistent[worst_id]
        self._persistent[eid] = flow

    def persistent_ids(self) -> set[str]:
        return set(self._persistent.keys())

    def is_persistent(self, eid: str) -> bool:
        return eid in self._persistent

    def snapshot(self) -> list[tuple[str, float]]:
        """Persistent members sorted by current flow, best first."""
        return sorted(self._persistent.items(), key=lambda kv: -kv[1])

    def reset(self) -> None:
        self._history.clear()
        self._persistent.clear()
