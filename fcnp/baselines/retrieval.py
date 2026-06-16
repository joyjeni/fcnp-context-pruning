"""Retrieval baselines: BM25 and dense cosine top-k."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from fcnp.baselines.base import BaselineResult
from fcnp.types import ContextElement


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


@dataclass
class BM25Baseline:
    """Okapi BM25 (Robertson & Walker, 1994) ranking over context elements.

    Standard parameters k1=1.5, b=0.75.
    """

    name: str = "BM25"
    k1: float = 1.5
    b: float = 0.75

    def _score(self, query_tokens: list[str], elements: list[ContextElement]) -> np.ndarray:
        docs = [_tokenize(e.text) for e in elements]
        N = len(docs)
        avgdl = sum(len(d) for d in docs) / max(N, 1)
        # Document frequencies
        df: dict[str, int] = {}
        for d in docs:
            for term in set(d):
                df[term] = df.get(term, 0) + 1

        scores = np.zeros(N)
        for i, d in enumerate(docs):
            dl = len(d)
            tf = Counter(d)
            s = 0.0
            for q in query_tokens:
                if q not in tf:
                    continue
                n_q = df.get(q, 0)
                idf = math.log(1 + (N - n_q + 0.5) / (n_q + 0.5))
                num = tf[q] * (self.k1 + 1)
                den = tf[q] + self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1e-9))
                s += idf * num / max(den, 1e-9)
            scores[i] = s
        return scores

    def compress(self, elements, query_text, query_embedding, keep_k) -> BaselineResult:
        t0 = time.perf_counter()
        q_tokens = _tokenize(query_text)
        scores = self._score(q_tokens, elements)
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


@dataclass
class DenseTopKBaseline:
    """Cosine similarity between dense embeddings and the query vector."""

    name: str = "DenseTopK"

    def compress(self, elements, query_text, query_embedding, keep_k) -> BaselineResult:
        t0 = time.perf_counter()
        if query_embedding is None or not elements:
            scores = np.zeros(len(elements))
        else:
            E = np.stack([e.embedding for e in elements]).astype(np.float64)
            En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
            q = query_embedding.astype(np.float64)
            q = q / (np.linalg.norm(q) + 1e-12)
            scores = En @ q
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
