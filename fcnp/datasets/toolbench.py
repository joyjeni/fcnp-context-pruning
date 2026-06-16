"""ToolBench dataset loader (OpenBMB ToolBench / ToolLLM).

Reference
---------
Qin, Y. et al. "ToolLLM: Facilitating Large Language Models to Master
16000+ Real-world APIs." ICLR 2024.

Each benchmark example contains:

    - query : natural-language user instruction
    - api_list : list of candidate API documents (50-200+ entries),
                 each with category_name / tool_name / api_name /
                 api_description / parameters
    - relevant_apis : ground-truth subset of api_list required to
                      answer the query (provided as
                      (tool_name, api_name) tuples)

For FCNP we treat each entry in api_list as a ContextElement. The
ground-truth relevant_apis set defines the evaluation oracle: a
correct compression must preserve these elements.

Splits available
----------------
    g1_instruction, g1_category, g1_tool,
    g2_instruction, g2_category,
    g3_instruction

G1 = single-tool, G2 = intra-category multi-tool, G3 = intra-collection
multi-tool.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from fcnp.types import ContextElement


@dataclass
class ToolBenchExample:
    query_id: str
    query: str
    api_list: list[dict]            # raw API docs
    relevant_apis: list[tuple[str, str]]  # (tool_name, api_name) pairs
    split: str = ""

    def relevant_keys(self) -> set[str]:
        return {f"{t}::{a}" for t, a in self.relevant_apis}

    def to_elements(self, embedder=None) -> list[ContextElement]:
        """Convert api_list into ContextElement objects.

        Parameters
        ----------
        embedder : callable(str) -> np.ndarray, optional
            Function that returns a dense embedding for a string. If
            None, falls back to a deterministic hashed embedding
            (suitable for CPU-only Kaggle benchmarking; replace with a
            real encoder for final results).
        """
        elements: list[ContextElement] = []
        for api in self.api_list:
            tool = str(api.get("tool_name", ""))
            name = str(api.get("api_name", ""))
            cat = str(api.get("category_name", ""))
            desc = str(api.get("api_description", "") or "")
            params = api.get("required_parameters", []) or []
            param_str = ", ".join(
                p.get("name", "") if isinstance(p, dict) else str(p) for p in params
            )
            text = (
                f"[{cat}] {tool} / {name}: {desc}"
                + (f" Parameters: {param_str}" if param_str else "")
            )
            key = f"{tool}::{name}"
            emb = embedder(text) if embedder else _hash_embedding(text, dim=128)
            elements.append(
                ContextElement(
                    id=key,
                    text=text,
                    embedding=emb,
                    importance=0.5,
                    citations=[key],
                    metadata={"tool": tool, "api": name, "category": cat},
                )
            )
        return elements


def _hash_embedding(text: str, dim: int = 128, seed: int = 1234) -> np.ndarray:
    """Deterministic feature-hashed embedding (CPU-friendly stand-in).

    Maps each whitespace token to a pseudo-random vector via a seeded
    hash; sums and L2-normalizes. Adequate for ranking experiments
    where the *relative* geometry matters more than absolute quality.
    """
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((4096, dim))
    vec = np.zeros(dim)
    for tok in text.lower().split():
        idx = (hash(tok) & 0xFFFFFFFF) % basis.shape[0]
        vec += basis[idx]
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


class ToolBenchLoader:
    """Loads ToolBench benchmark splits from HuggingFace or local JSON.

    Two loading modes:

    1. HuggingFace ``datasets`` library (preferred on Kaggle):

           loader = ToolBenchLoader.from_huggingface(
               repo="tuandunghcmut/toolbench-v1", config="benchmark"
           )
           for ex in loader.iter_split("g1_instruction", limit=200):
               ...

    2. Local OpenBMB ToolBench JSON files (mirror of the official repo):

           loader = ToolBenchLoader.from_local("/path/to/ToolBench/data")
           for ex in loader.iter_split("G1_instruction", limit=200):
               ...
    """

    def __init__(self, source: str, mode: str):
        self.source = source
        self.mode = mode  # "hf" | "local"
        self._hf_ds = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_huggingface(
        cls,
        repo: str = "tuandunghcmut/toolbench-v1",
        config: str = "benchmark",
    ) -> "ToolBenchLoader":
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError(
                "pip install datasets  # required for HuggingFace loading"
            ) from e
        loader = cls(source=f"hf://{repo}@{config}", mode="hf")
        loader._hf_ds = load_dataset(repo, config)
        return loader

    @classmethod
    def from_local(cls, root: str) -> "ToolBenchLoader":
        if not Path(root).exists():
            raise FileNotFoundError(root)
        return cls(source=str(root), mode="local")

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------
    def iter_split(
        self,
        split: str,
        limit: int | None = None,
    ) -> Iterator[ToolBenchExample]:
        if self.mode == "hf":
            yield from self._iter_hf(split, limit)
        else:
            yield from self._iter_local(split, limit)

    def _iter_hf(self, split: str, limit: int | None) -> Iterator[ToolBenchExample]:
        ds = self._hf_ds[split]
        n = len(ds) if limit is None else min(limit, len(ds))
        for i in range(n):
            row = ds[i]
            api_list = _parse_json_field(row.get("api_list"))
            rel = _parse_json_field(row.get("relevant_apis"))
            yield ToolBenchExample(
                query_id=str(row.get("query_id", i)),
                query=str(row.get("query", "")),
                api_list=api_list if isinstance(api_list, list) else [],
                relevant_apis=_normalize_relevant(rel),
                split=split,
            )

    def _iter_local(self, split: str, limit: int | None) -> Iterator[ToolBenchExample]:
        path = Path(self.source) / "test_instruction" / f"{split}.json"
        if not path.exists():
            raise FileNotFoundError(f"Expected ToolBench split file at {path}")
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        if limit is not None:
            data = data[:limit]
        for i, row in enumerate(data):
            yield ToolBenchExample(
                query_id=str(row.get("query_id", i)),
                query=str(row.get("query", "")),
                api_list=row.get("api_list", []) or [],
                relevant_apis=_normalize_relevant(
                    row.get("relevant_apis") or row.get("relevant APIs")
                ),
                split=split,
            )


def _parse_json_field(v):
    if v is None:
        return None
    if isinstance(v, (list, dict)):
        return v
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return None


def _normalize_relevant(rel) -> list[tuple[str, str]]:
    if rel is None:
        return []
    out = []
    for r in rel:
        if isinstance(r, dict):
            t = r.get("tool_name") or r.get("tool") or ""
            a = r.get("api_name") or r.get("api") or ""
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            t, a = r[0], r[1]
        else:
            continue
        out.append((str(t), str(a)))
    return out


def load_toolbench(
    source: str = "huggingface",
    repo: str = "tuandunghcmut/toolbench-v1",
    root: str | None = None,
    splits: Iterable[str] | None = None,
    limit_per_split: int | None = 200,
) -> dict[str, list[ToolBenchExample]]:
    """Convenience: load multiple splits at once.

    Returns
    -------
    dict[split_name, list[ToolBenchExample]]
    """
    if source == "huggingface":
        loader = ToolBenchLoader.from_huggingface(repo=repo)
        default_splits = [
            "g1_instruction", "g1_category", "g1_tool",
            "g2_instruction", "g2_category",
            "g3_instruction",
        ]
    elif source == "local":
        assert root, "root path required for local loading"
        loader = ToolBenchLoader.from_local(root)
        default_splits = [
            "G1_instruction", "G1_category", "G1_tool",
            "G2_instruction", "G2_category",
            "G3_instruction",
        ]
    else:
        raise ValueError(f"unknown source: {source}")

    splits = list(splits) if splits else default_splits
    out: dict[str, list[ToolBenchExample]] = {}
    for s in splits:
        try:
            out[s] = list(loader.iter_split(s, limit=limit_per_split))
        except (FileNotFoundError, KeyError) as e:
            print(f"[warn] skipping split {s}: {e}")
    return out
