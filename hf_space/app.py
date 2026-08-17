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

import time

import gradio as gr
import numpy as np
import plotly.graph_objects as go

from fcnp.pruner import FlowBasedNetworkPruner, FCNPConfig
from fcnp.types import ContextElement

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


def run_comparison(query: str, candidates_raw: str, keep_fraction: float):
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
    )
    pruner = FlowBasedNetworkPruner(cfg)
    t0 = time.perf_counter()
    result = pruner.prune(elements, query_embedding=query_vec)
    t_fcnp = (time.perf_counter() - t0) * 1000
    flow_order = list(np.argsort(-result.node_flow))[:k]
    fcnp_ids = {ids[i] for i in flow_order}
    kept_by_method["FCNP"] = fcnp_ids
    rows.append(
        [
            "FCNP",
            ", ".join(sorted(fcnp_ids)),
            f"{n / max(k, 1):.2f}x",
            f"{t_fcnp:.2f} ms",
            "converged" if result.converged else f"stopped @ {result.iterations}",
        ]
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

    return rows, note, flow_chart


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

with gr.Blocks(title="FCNP Context Pruning Demo") as demo:
    gr.Markdown(
        "# FCNP — Flow-Based Context Network Pruning\n"
        "Paste a query and candidate documents/APIs (`id | text` per line, id optional). "
        "Compare what FCNP's Physarum-style graph-flow selection keeps vs. BM25, dense "
        "top-k cosine similarity, and random selection, at the same keep-fraction.\n\n"
        "This demo shows *mechanism*, not accuracy — there's no ground-truth label for "
        "freeform text. For the ground-truth ToolBench benchmark numbers (where FCNP "
        "currently does **not** beat dense top-k on F1, see the "
        "[GitHub README](https://github.com/joyjeni/fcnp-context-pruning#readme))."
    )
    with gr.Row():
        with gr.Column(scale=1):
            query_in = gr.Textbox(label="Query", value=EXAMPLE_QUERY, lines=2)
            cands_in = gr.Textbox(
                label="Candidates (one per line, 'id | text')",
                value=EXAMPLE_CANDIDATES,
                lines=12,
            )
            frac_in = gr.Slider(
                label="Keep fraction", minimum=0.1, maximum=0.9, value=0.2, step=0.05
            )
            run_btn = gr.Button("Run comparison", variant="primary")
        with gr.Column(scale=1):
            table_out = gr.Dataframe(
                headers=["Method", "Kept IDs", "Compression", "Latency", "FCNP status"],
                label="Retained set per method",
            )
            note_out = gr.Markdown()
            flow_out = gr.Plot(label="FCNP node flow scores")

    run_btn.click(
        run_comparison,
        inputs=[query_in, cands_in, frac_in],
        outputs=[table_out, note_out, flow_out],
    )

if __name__ == "__main__":
    demo.launch()
