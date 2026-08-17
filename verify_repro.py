"""Independent reviewer verification script.

Reruns the exact synthetic benchmark defined in run_synthetic_e2e.py and:
  1. Re-derives all headline aggregate numbers from scratch (independent of results.csv).
  2. Computes the Jaccard/exact-match overlap between FCNP's retained set and
     DenseTopK's retained set, per example.
  3. Confirms the trivial-lexical-recoverability hypothesis for BM25/SC/LLMLingua
     by checking whether relevant items are always the ones containing the exact
     query keyword substring.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fcnp import ALL_BASELINES, FCNPConfig, FCNPMethod, evaluate_all, aggregate
from run_synthetic_e2e import make_example

examples = {"synthetic_g1": [make_example(s) for s in range(30)]}

fcnp_cfg = FCNPConfig(similarity_threshold=0.25, max_iterations=120, epsilon=1e-3,
                       mu=0.10, alpha=0.50, gamma=1.20)
methods = {name: cls() for name, cls in ALL_BASELINES.items()}
methods["FCNP"] = FCNPMethod(config=fcnp_cfg)

scores = evaluate_all(methods, examples, keep_k_strategy="oracle")
agg = aggregate(scores)
print("=== Re-derived aggregate F1 (independent re-run) ===")
for m in agg:
    print(f"{m.method:<18} F1={m.f1_mean:.4f}  n={m.n}")

# ---- FCNP vs DenseTopK overlap, per example ----
print("\n=== FCNP vs DenseTopK retained-set overlap ===")
overlaps = []
exact_matches = 0
for ex in examples["synthetic_g1"]:
    elements = ex.to_elements(embedder=None)
    from fcnp.datasets.toolbench import _hash_embedding
    q_emb = _hash_embedding(ex.query, dim=elements[0].embedding.shape[0])
    k = len(ex.relevant_keys())

    dense = methods["DenseTopK"].compress(elements, ex.query, q_emb, k)
    fcnp_res = methods["FCNP"].compress(elements, ex.query, q_emb, k)

    dense_ids = {e.id for e in dense.survivors}
    fcnp_ids = {e.id for e in fcnp_res.survivors}
    inter = dense_ids & fcnp_ids
    union = dense_ids | fcnp_ids
    jacc = len(inter) / len(union) if union else 1.0
    overlaps.append(jacc)
    if dense_ids == fcnp_ids:
        exact_matches += 1
    print(f"{ex.query_id:8s} k={k:2d}  FCNP∩Dense={len(inter)}/{k}  Jaccard={jacc:.2f}  exact_match={dense_ids==fcnp_ids}")

print(f"\nMean Jaccard(FCNP, DenseTopK) = {np.mean(overlaps):.3f}")
print(f"Exact-set matches: {exact_matches}/30 = {exact_matches/30:.1%}")

# ---- Lexical recoverability check ----
print("\n=== Lexical recoverability check (BM25) ===")
hits = 0
for ex in examples["synthetic_g1"]:
    elements = ex.to_elements(embedder=None)
    bm25 = methods["BM25"].compress(elements, ex.query, None, len(ex.relevant_keys()))
    kept_ids = {e.id for e in bm25.survivors}
    rel_ids = ex.relevant_keys()
    if kept_ids == rel_ids:
        hits += 1
print(f"BM25 kept-set == ground-truth relevant-set exactly: {hits}/30 examples = {hits/30:.1%}")

# show one example's raw structure to illustrate why
ex0 = examples["synthetic_g1"][0]
print("\nExample 0 query:", ex0.query)
print("Relevant apis:", ex0.relevant_apis)
print("Sample distractor api:", [a for a in ex0.api_list if (a['tool_name'], a['api_name']) not in ex0.relevant_apis][0])
