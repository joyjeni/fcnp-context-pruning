"""Baseline compression methods for comparison with FCNP.

Implements seven baselines commonly used as reference points in the
prompt-compression literature:

    - NoCompression : full context (upper bound on accuracy)
    - Random        : uniform random keep (lower bound)
    - TopKImportance: keep highest-importance elements (greedy)
    - BM25Retrieval : Robertson-Sparck-Jones BM25 against the query
    - DenseTopK     : cosine top-k against query embedding
    - SelectiveContextProxy : self-information ranking via TF-IDF entropy
                              (Li et al., EMNLP 2023, *Selective Context*)
    - LLMLinguaProxy        : token-/element-level importance scoring
                              (Jiang et al., EMNLP 2023, *LLMLingua*)

The "proxy" baselines reproduce the *ranking principle* of each method
without requiring a small LLM for perplexity scoring (which is
prohibitive in a CPU-only Kaggle environment). The full LLM-based
versions can be plugged in by swapping the scoring function.
"""

from fcnp.baselines.base import Baseline, BaselineResult
from fcnp.baselines.simple import (
    NoCompressionBaseline,
    RandomBaseline,
    TopKImportanceBaseline,
)
from fcnp.baselines.retrieval import BM25Baseline, DenseTopKBaseline
from fcnp.baselines.selective_context import SelectiveContextProxyBaseline
from fcnp.baselines.llmlingua import LLMLinguaProxyBaseline

ALL_BASELINES = {
    "NoCompression": NoCompressionBaseline,
    "Random": RandomBaseline,
    "TopKImportance": TopKImportanceBaseline,
    "BM25": BM25Baseline,
    "DenseTopK": DenseTopKBaseline,
    "SelectiveContext": SelectiveContextProxyBaseline,
    "LLMLingua": LLMLinguaProxyBaseline,
}

__all__ = [
    "Baseline",
    "BaselineResult",
    "NoCompressionBaseline",
    "RandomBaseline",
    "TopKImportanceBaseline",
    "BM25Baseline",
    "DenseTopKBaseline",
    "SelectiveContextProxyBaseline",
    "LLMLinguaProxyBaseline",
    "ALL_BASELINES",
]
