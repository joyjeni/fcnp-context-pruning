"""Lightweight extractive summarizer for the SUMMARIZE tier.

Improvement #2 uses this as the default, dependency-free summarizer so
FCNP keeps working with zero extra model calls out of the box. Any
downstream integration can instead pass a real LLM-backed
``summarizer: Callable[[str, str], str]`` (text, query) -> summary into
``FlowBasedNetworkPruner.prune(..., summarizer=...)`` — this module is
just the sane default, not a hard requirement.

The extractive approach: score sentences by unigram overlap with the
query (cheap TF proxy for relevance) plus a mild position prior (leads
and conclusions tend to carry the claim), then keep the
highest-scoring sentences until a token budget is hit.
"""

from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def extractive_summarize(
    text: str,
    query: str | None = None,
    max_tokens: int = 40,
) -> str:
    """Return a compacted version of ``text`` targeting ``max_tokens`` words.

    Falls back to simple truncation for very short inputs where sentence
    splitting would not help.
    """
    text = text.strip()
    if not text:
        return text

    words = text.split()
    if len(words) <= max_tokens:
        return text

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sentences) <= 1:
        return " ".join(words[:max_tokens]) + " …"

    query_terms = set(_tokenize(query)) if query else set()

    scored: list[tuple[float, int, str]] = []
    for i, sent in enumerate(sentences):
        terms = _tokenize(sent)
        if not terms:
            continue
        overlap = len(set(terms) & query_terms)
        position_bonus = 0.15 if i == 0 else (0.10 if i == len(sentences) - 1 else 0.0)
        density = overlap / max(len(terms), 1)
        score = density + position_bonus
        scored.append((score, i, sent))

    scored.sort(key=lambda t: (-t[0], t[1]))

    kept: list[tuple[int, str]] = []
    budget = max_tokens
    for score, idx, sent in scored:
        n_words = len(sent.split())
        if budget - n_words < 0 and kept:
            continue
        kept.append((idx, sent))
        budget -= n_words
        if budget <= 0:
            break

    kept.sort(key=lambda t: t[0])
    summary = " ".join(s for _, s in kept)
    return summary if summary else " ".join(words[:max_tokens]) + " …"
