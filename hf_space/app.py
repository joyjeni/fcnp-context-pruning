"""FCNP interactive demo — Hugging Face Space (Gradio).

Lets a visitor paste a query + a list of candidate documents/APIs and compare
FCNP's graph-flow selection against BM25, dense top-k cosine, and random
baselines: which items get kept, the FCNP conductance/flow scores, and
per-method latency.

This is a *mechanism explorer*, not an accuracy benchmark — there is no
ground-truth relevance label for freeform pasted text, so no F1/precision is
shown here. For the honest, ground-truth benchmark numbers (FCNP vs 7
baselines on ToolBench), see the project README and the live Vercel
dashboard linked in this Space's README.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import spaces

from fcnp.pruner import FlowBasedNetworkPruner, FCNPConfig
from fcnp.types import ContextElement, Tier
from fcnp.data_gov_agri import (
    RESOURCE_CITATION,
    RESOURCE_URL,
    load_snapshot,
    snapshot_to_candidate_lines,
)
from fcnp.cost import cost_comparison_table, format_markdown_table

_SNAPSHOT_PATH = Path(__file__).parent / "data" / "agri_mandi_snapshot.json"
_AGRI_SNAPSHOT = None


def _get_agri_snapshot():
    """Load the real data.gov.in mandi-price snapshot bundled with this Space.

    Prefers a live fetch when ``DATA_GOV_API_KEY`` is set as an HF Space
    secret; otherwise falls back to the static snapshot committed at
    deploy time (see fcnp/data_gov_agri.py for the fetch/fallback logic).
    """
    global _AGRI_SNAPSHOT
    if _AGRI_SNAPSHOT is not None:
        return _AGRI_SNAPSHOT
    api_key = os.environ.get("DATA_GOV_API_KEY", "")
    if api_key:
        try:
            from fcnp.data_gov_agri import fetch_snapshot_live

            _AGRI_SNAPSHOT = fetch_snapshot_live(api_key)
            return _AGRI_SNAPSHOT
        except Exception:
            pass  # fall through to bundled snapshot
    _AGRI_SNAPSHOT = load_snapshot(_SNAPSHOT_PATH)
    return _AGRI_SNAPSHOT

_ENCODER = None


def _get_encoder():
    global _ENCODER
    if _ENCODER is None:
        from sentence_transformers import SentenceTransformer

        _ENCODER = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _ENCODER


def _embed(texts: list[str]) -> np.ndarray:
    enc = _get_encoder()
    return enc.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(
        np.float32
    )


def _parse_candidates(raw: str) -> list[tuple[str, str]]:
    """Parse 'id | text' lines (id optional -> cand_0, cand_1, ...)."""
    out = []
    for i, line in enumerate(l.strip() for l in raw.splitlines()):
        if not line:
            continue
        if "|" in line:
            cid, text = line.split("|", 1)
            out.append((cid.strip() or f"cand_{i}", text.strip()))
        else:
            out.append((f"cand_{i}", line))
    return out


def _bm25_rank(query: str, docs: list[str]) -> list[int]:
    from rank_bm25 import BM25Okapi

    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    return list(np.argsort(-scores))


def _dense_rank(query_vec: np.ndarray, doc_vecs: np.ndarray) -> list[int]:
    sims = doc_vecs @ query_vec
    return list(np.argsort(-sims))


@spaces.GPU
def run_comparison(query: str, candidates_raw: str, keep_fraction: float, enable_hybrid: bool):
    parsed = _parse_candidates(candidates_raw)
    if not query.strip() or len(parsed) < 2:
        raise gr.Error("Provide a query and at least 2 candidate lines.")

    ids = [c[0] for c in parsed]
    texts = [c[1] for c in parsed]
    n = len(texts)
    k = max(1, round(n * keep_fraction))

    all_vecs = _embed([query] + texts)
    query_vec, doc_vecs = all_vecs[0], all_vecs[1:]

    rows = []
    kept_by_method: dict[str, set[str]] = {}

    # --- FCNP ---
    elements = [
        ContextElement(id=ids[i], text=texts[i], embedding=doc_vecs[i])
        for i in range(n)
    ]
    cfg = FCNPConfig(
        similarity_threshold=0.25,
        epsilon=1e-3,
        max_iterations=200,
        mu=0.10,
        alpha=0.50,
        gamma=1.20,
        keep_top_k_fraction=keep_fraction,
        summarize_top_k_fraction=0.20,
        enable_hybrid_tiering=enable_hybrid,
    )
    pruner = FlowBasedNetworkPruner(cfg)
    t0 = time.perf_counter()
    result = pruner.prune(elements, query_embedding=query_vec, query_text=query)
    t_fcnp = (time.perf_counter() - t0) * 1000
    fcnp_ids = {e.id for e in result.survivors}
    kept_by_method["FCNP"] = fcnp_ids
    status = "converged" if result.converged else f"stopped @ {result.iterations}"
    if enable_hybrid:
        tier_summary = ", ".join(f"{k2}={v}" for k2, v in sorted(result.tier_counts.items()))
        status = f"{status} | tiers: {tier_summary}"
    rows.append(
        [
            "FCNP" + ("-Hybrid" if enable_hybrid else ""),
            ", ".join(sorted(fcnp_ids)),
            f"{result.compression_ratio:.2f}x",
            f"{t_fcnp:.2f} ms",
            status,
        ]
    )

    tier_lines = []
    if enable_hybrid:
        for e in sorted(result.survivors, key=lambda x: -(x.importance or 0)):
            tag = e.tier.value if e.tier else "?"
            shown_text = e.output_text()
            tier_lines.append(f"- **[{tag}]** `{e.id}` — {shown_text[:140]}")
    tier_panel = (
        "### Hybrid tier assignment\n" + "\n".join(tier_lines)
        if tier_lines
        else "_Enable \"hybrid tiering\" above to see per-item keep/summarize/drop/persistent tags._"
    )

    # --- Dense top-k cosine ---
    t0 = time.perf_counter()
    dense_order = _dense_rank(query_vec, doc_vecs)[:k]
    t_dense = (time.perf_counter() - t0) * 1000
    dense_ids = {ids[i] for i in dense_order}
    kept_by_method["DenseTopK"] = dense_ids
    rows.append(
        ["DenseTopK", ", ".join(sorted(dense_ids)), f"{n / max(k, 1):.2f}x", f"{t_dense:.2f} ms", "—"]
    )

    # --- BM25 ---
    t0 = time.perf_counter()
    bm25_order = _bm25_rank(query, texts)[:k]
    t_bm25 = (time.perf_counter() - t0) * 1000
    bm25_ids = {ids[i] for i in bm25_order}
    kept_by_method["BM25"] = bm25_ids
    rows.append(
        ["BM25", ", ".join(sorted(bm25_ids)), f"{n / max(k, 1):.2f}x", f"{t_bm25:.2f} ms", "—"]
    )

    # --- Random ---
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    rand_order = rng.choice(n, size=k, replace=False)
    t_rand = (time.perf_counter() - t0) * 1000
    rand_ids = {ids[i] for i in rand_order}
    kept_by_method["Random"] = rand_ids
    rows.append(
        ["Random", ", ".join(sorted(rand_ids)), f"{n / max(k, 1):.2f}x", f"{t_rand:.2f} ms", "—"]
    )

    overlap_dense = len(fcnp_ids & dense_ids) / max(len(fcnp_ids | dense_ids), 1)
    note = (
        f"FCNP \u2229 DenseTopK Jaccard overlap: **{overlap_dense:.2f}** "
        f"(on the published ToolBench benchmark this overlap is ~0.98 — the two "
        f"methods usually agree, and DenseTopK is ~100x faster; see the README "
        f"for the full honest comparison)."
    )

    flow_chart = go.Figure(
        data=[go.Bar(x=ids, y=[float(v) for v in result.node_flow])]
    )
    flow_chart.update_layout(
        title="FCNP node flow score (higher = kept)",
        xaxis_title="candidate id",
        yaxis_title="aggregate flow",
    )

    cost_note = ""
    if enable_hybrid:
        n_summarized = result.tier_counts.get("summarize", 0)
        comp = cost_comparison_table(fcnp_latencies_ms=[t_fcnp], n_reprune_events=max(1, n_summarized))
        cost_note = (
            "\n\n### Compute-cost vs token-cost (improvement #5)\n"
            + format_markdown_table(comp)
            + "\n\n_FCNP's summarization step above used the built-in "
            "dependency-free extractive summarizer — zero LLM calls. The "
            "table shows what an LLM-driven compressor would have cost "
            "instead for the same number of compression events "
            f"({max(1, n_summarized)})."
        )

    return rows, note + cost_note, flow_chart, tier_panel


EXAMPLE_QUERY = "Help me check the weather and book a flight for my trip to Bangalore."
EXAMPLE_CANDIDATES = """weather_tool | Get current weather conditions for a city.
get_weather_data | Retrieve hourly weather forecast by coordinates.
flight_tool | Search and book flights between two airports.
get_flight_data | Retrieve flight status and gate information.
stock_distractor_1 | Get real-time stock price for a ticker symbol.
music_distractor_2 | Search and stream songs by artist name.
recipe_distractor_3 | Find recipes by ingredient list.
calendar_distractor_4 | Create and manage calendar events.
translate_distractor_5 | Translate text between languages.
news_distractor_6 | Fetch top news headlines by category."""

AGRI_EXAMPLE_QUERY = "Which mandi has the best tomato and onion prices in Karnataka today?"


def _agri_example_candidates() -> str:
    snap = _get_agri_snapshot()
    lines = snapshot_to_candidate_lines(snap, limit=25)
    return "\n".join(f"{cid} | {text}" for cid, text in lines)


def load_example_set(choice: str):
    if choice == "Real Agri mandi prices (data.gov.in)":
        snap = _get_agri_snapshot()
        fallback_note = (
            f"_Source: [{RESOURCE_CITATION}]({RESOURCE_URL}) — fetched {snap.fetched_at}. "
            f"{snap.notes}_"
        )
        return AGRI_EXAMPLE_QUERY, _agri_example_candidates(), fallback_note
    return EXAMPLE_QUERY, EXAMPLE_CANDIDATES, "_Synthetic weather/flight-tool example set._"

with gr.Blocks(title="FCNP Context Pruning Demo") as demo:
    gr.Markdown(
        "# FCNP — Flow-Based Context Network Pruning\n"
        "Paste a query and candidate documents/APIs (`id | text` per line, id optional). "
        "Compare what FCNP's Physarum-style graph-flow selection keeps vs. BM25, dense "
        "top-k cosine similarity, and random selection, at the same keep-fraction.\n\n"
        "This demo shows *mechanism*, not accuracy — there's no ground-truth label for "
        "freeform text. For the ground-truth ToolBench benchmark numbers (where FCNP "
        "currently does **not** beat dense top-k on F1, see the "
        "[GitHub README](https://github.com/joyjeni/fcnp-context-pruning#readme)).\n\n"
        "**New:** toggle *hybrid tiering* to see improvements #1–#3 in action — medium-flow "
        "items get summarized instead of dropped, high-flow items are force-kept across "
        "rounds via a persistent-memory tier, and re-pruning is only triggered when the "
        "context's flow-entropy actually drifts. Switch the example set to real Karnataka "
        "mandi (market) prices from data.gov.in, in the same catalog style as the sibling "
        "[Session-Aware ToolBench Rerank](https://github.com/joyjeni/session-aware-toolbench-rerank) "
        "project's Agri extension."
    )
    with gr.Row():
        with gr.Column(scale=1):
            example_set = gr.Radio(
                label="Example set",
                choices=["Synthetic (weather/flight tools)", "Real Agri mandi prices (data.gov.in)"],
                value="Synthetic (weather/flight tools)",
            )
            source_note = gr.Markdown("_Synthetic weather/flight-tool example set._")
            query_in = gr.Textbox(label="Query", value=EXAMPLE_QUERY, lines=2)
            cands_in = gr.Textbox(
                label="Candidates (one per line, 'id | text')",
                value=EXAMPLE_CANDIDATES,
                lines=12,
            )
            frac_in = gr.Slider(
                label="Keep fraction", minimum=0.1, maximum=0.9, value=0.2, step=0.05
            )
            hybrid_in = gr.Checkbox(
                label="Enable hybrid tiering (improvements #1–#3: dynamic trigger, "
                "keep/summarize/drop/persistent tiers)",
                value=False,
            )
            run_btn = gr.Button("Run comparison", variant="primary")
        with gr.Column(scale=1):
            table_out = gr.Dataframe(
                headers=["Method", "Kept IDs", "Compression", "Latency", "FCNP status"],
                label="Retained set per method",
            )
            note_out = gr.Markdown()
            flow_out = gr.Plot(label="FCNP node flow scores")
            tier_out = gr.Markdown()

    example_set.change(
        load_example_set,
        inputs=[example_set],
        outputs=[query_in, cands_in, source_note],
    )

    run_btn.click(
        run_comparison,
        inputs=[query_in, cands_in, frac_in, hybrid_in],
        outputs=[table_out, note_out, flow_out, tier_out],
    )

if __name__ == "__main__":
    demo.launch()
