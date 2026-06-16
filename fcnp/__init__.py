"""FCNP — Flow-Based Context Network Pruning."""

from fcnp.types import ContextElement, PruneResult
from fcnp.pruner import FlowBasedNetworkPruner, FCNPConfig
from fcnp.metrics import BenchmarkMetrics, evaluate

from fcnp.baselines import ALL_BASELINES
from fcnp.baselines.fcnp_wrapper import FCNPMethod
from fcnp.datasets.toolbench import (
    ToolBenchExample,
    ToolBenchLoader,
    load_toolbench,
)
from fcnp.eval import (
    ExampleScore,
    MethodAggregate,
    evaluate_all,
    aggregate,
    pairwise_significance,
    bootstrap_ci,
    wilcoxon_paired,
)
from fcnp.report import write_report

__version__ = "0.1.0"

__all__ = [
    "ContextElement",
    "PruneResult",
    "FlowBasedNetworkPruner",
    "FCNPConfig",
    "FCNPMethod",
    "BenchmarkMetrics",
    "evaluate",
    "ALL_BASELINES",
    "ToolBenchExample",
    "ToolBenchLoader",
    "load_toolbench",
    "ExampleScore",
    "MethodAggregate",
    "evaluate_all",
    "aggregate",
    "pairwise_significance",
    "bootstrap_ci",
    "wilcoxon_paired",
    "write_report",
]
