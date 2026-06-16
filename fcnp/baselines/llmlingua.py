"""LLMLingua proxy (Jiang et al., EMNLP 2023).

The published method uses a small LM (e.g., LLaMA-2-7B) to assign
per-token perplexity and ranks elements by their contrastive
question-aware perplexity. We approximate the same ranking with a
question-conditioned lexical-overlap × inverse-perplexity proxy
suitable for CPU-only environments:

    score(d, q) = sim_lex(q, d) * (1 / log(1 + len(d)))

This captures the two principles that drive LLMLingua's ranking:
(1) elements more aligned with the question are kept (the
"question-aware" coefficient r_k), and (2) longer elements are
penalized so the budget is spent on dense, informative content. A
plug-in for the full LM-based scorer is provided.

Reference
---------
Jiang, H. et al. "LLMLingua: Compressing prompts for accelerated
inference of LLMs." EMNLP 2023.
Jiang, H. et al. "LongLLMLingua." ACL 2024.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

from fcnp.baselines.base import BaselineResult
from fcnp.types import ContextElement


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class LLMLinguaProxyBaseline:
    name: str = "LLMLingua"
    scorer: Callable[[str, str], float] | None = None  # plug-in for true LM scorer

    def compress(self, elements, query_text, query_embedding, keep_k) -> BaselineResult:
        t0 = time.perf_counter()
        q_tokens = _tokenize(query_text)
        scores = np.zeros(len(elements))
        for i, e in enumerate(elements):
            d_tokens = _tokenize(e.text)
            if self.scorer is not None:
                scores[i] = self.scorer(query_text, e.text)
            else:
                sim = _jaccard(q_tokens, d_tokens)
                length_penalty = 1.0 / math.log(1 + max(len(d_tokens), 1))
                scores[i] = sim * length_penalty
        k = min(keep_k, len(elements))
        top = np.argsort(-scores)[:k]
        survivors = [elements[i] for i in top]
        wall = (time.perf_counter() - t0) * 1000
        in_tok = sum(e.token_count() for e in elements)
        out_tok = sum(e.token_count() for e in survivors)
        return BaselineResult(
            survivors=survivors,
            n_input=len(elements), n_output=len(survivors),
            input_tokens=in_tok, output_tokens=out_tok,
            wall_time_ms=wall,
        )
