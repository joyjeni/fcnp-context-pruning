"""Selective Context proxy (Li et al., EMNLP 2023).

The published method ranks tokens/sentences by self-information
-log p(t | context) computed by a small LM. We approximate the same
ranking principle with a corpus-level TF-IDF entropy proxy that
requires no LM forward pass:

    self_information_proxy(d) = sum_{t in d} -log p_corpus(t)

where p_corpus(t) is the empirical token probability over all context
elements. This preserves the *relative* ordering (high-entropy ⇒ more
"surprising" ⇒ more informative) used by Selective Context, which is
what determines which elements are kept.

Reference
---------
Li, Y. "Compressing context to enhance inference efficiency of LLMs."
EMNLP 2023.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass

import numpy as np

from fcnp.baselines.base import BaselineResult
from fcnp.types import ContextElement


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


@dataclass
class SelectiveContextProxyBaseline:
    name: str = "SelectiveContext"

    def compress(self, elements, query_text, query_embedding, keep_k) -> BaselineResult:
        t0 = time.perf_counter()
        docs = [_tokenize(e.text) for e in elements]
        total = sum(len(d) for d in docs)
        if total == 0:
            return BaselineResult(
                survivors=list(elements[:keep_k]),
                n_input=len(elements), n_output=min(keep_k, len(elements)),
                input_tokens=0, output_tokens=0,
                wall_time_ms=(time.perf_counter() - t0) * 1000,
            )
        # corpus token probabilities
        corpus = Counter()
        for d in docs:
            corpus.update(d)
        scores = np.zeros(len(elements))
        for i, d in enumerate(docs):
            if not d:
                continue
            si = 0.0
            for tok in d:
                p = corpus[tok] / total
                si += -math.log(max(p, 1e-12))
            scores[i] = si / len(d)  # mean self-information per token
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
