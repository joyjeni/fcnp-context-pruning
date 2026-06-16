"""Benchmark metrics for FCNP."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fcnp.pruner import FCNPConfig, FlowBasedNetworkPruner
from fcnp.types import ContextElement, PruneResult


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a single benchmark run."""

    run_id: str
    timestamp: float
    n_input_elements: int
    n_output_elements: int
    input_tokens: int
    output_tokens: int
    compression_ratio: float
    reduction_pct: float
    citation_accuracy: float          # 0..1, fraction of citations preserved
    iterations: int
    converged: bool
    wall_time_ms: float
    config: dict = field(default_factory=dict)
    flow_distribution: list[float] = field(default_factory=list)  # node_flow sample
    per_dataset: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p


def citation_accuracy(
    original: list[ContextElement],
    survivors: list[ContextElement],
    required_citations: set[str] | None = None,
) -> float:
    """Fraction of required citations preserved after pruning.

    If ``required_citations`` is provided, accuracy is computed against
    that set. Otherwise it is computed against the union of citations
    present in ``original``.
    """
    if required_citations is None:
        required_citations = {c for e in original for c in e.citations}
    if not required_citations:
        return 1.0
    preserved = {c for e in survivors for c in e.citations}
    return len(required_citations & preserved) / len(required_citations)


def evaluate(
    elements: list[ContextElement],
    query_embedding: np.ndarray | None = None,
    config: FCNPConfig | None = None,
    run_id: str | None = None,
    required_citations: set[str] | None = None,
) -> BenchmarkMetrics:
    """Run FCNP once and return a populated BenchmarkMetrics object."""
    cfg = config or FCNPConfig()
    pruner = FlowBasedNetworkPruner(cfg)

    t0 = time.perf_counter()
    result: PruneResult = pruner.prune(elements, query_embedding=query_embedding)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    cite_acc = citation_accuracy(elements, result.survivors, required_citations)

    # Sample flow distribution (cap at 256 points for JSON size)
    flow = result.node_flow
    if flow.size > 256:
        idx = np.linspace(0, flow.size - 1, 256).astype(int)
        flow_sample = flow[idx].tolist()
    else:
        flow_sample = flow.tolist()

    return BenchmarkMetrics(
        run_id=run_id or f"run-{int(time.time())}",
        timestamp=time.time(),
        n_input_elements=result.n_input,
        n_output_elements=result.n_output,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        compression_ratio=result.compression_ratio,
        reduction_pct=result.reduction_pct,
        citation_accuracy=cite_acc,
        iterations=result.iterations,
        converged=result.converged,
        wall_time_ms=wall_ms,
        config=asdict(cfg),
        flow_distribution=flow_sample,
    )
