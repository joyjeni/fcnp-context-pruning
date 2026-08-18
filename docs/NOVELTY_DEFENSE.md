# Literature Grounding & Novelty Defense — FCNP (PhD Objective 4)

**Purpose:** This document exists to be defended against, not just read. For every contribution claimed in this repository it lists (a) the strongest real, verifiable prior-art it should be compared against, (b) whether that prior art already implements the exact mechanism, and (c) a one-line novelty verdict. Every citation below was located via live search and traces to a real URL/DOI — none was generated from memory alone. Where a claim could not be independently verified against a primary academic source, that gap is stated explicitly rather than filled in.

Last verified: 2026-08-18.

---

## 1. Core contribution — Kirchhoff/Physarum flow-network pruning applied to LLM context compression

FCNP models context chunks as nodes in a conductance network, solves the Kirchhoff linear system \\(L(D)\\mathbf{p} = \\mathbf{I}\\), computes edge flow \\(Q_{ij} = D_{ij}|p_i - p_j|\\), and updates conductance via the Physarum rule \\(D_{ij}(t+1) = (1-\\mu)D_{ij} + \\alpha|Q_{ij}|^\\gamma\\) — using this joint, physically-coupled computation to decide keep/prune status for **all** chunks simultaneously, rather than ranking chunks independently the way prior compression methods do.

**Mechanism source (established, peer-reviewed):**

| Paper | Venue / Year | Link |
|---|---|---|
| Tero, A. et al. "Rules for Biologically Inspired Adaptive Network Design." | *Science*, vol. 327, no. 5964, pp. 439–442 (2010) | https://www.science.org/doi/10.1126/science.1177894 |
| Bonifaci, V., Mehlhorn, K., Varma, G. "Physarum can compute shortest paths." | *Journal of Theoretical Biology*, vol. 309, pp. 121–133 (2012). DOI `10.1016/j.jtbi.2012.06.017` | https://pubmed.ncbi.nlm.nih.gov/22732274/ (arXiv preprint version: https://arxiv.org/abs/1106.0423) |
| "Physarum Computations" (survey) | STACS 2013 (LIPIcs, peer-reviewed) | https://drops.dagstuhl.de/storage/00lipics/lipics-vol020-stacs2013/LIPIcs.STACS.2013.5/LIPIcs.STACS.2013.5.pdf |

**Baselines FCNP is positioned against (independent-ranking compression):**

| Paper | Venue / Year | Link |
|---|---|---|
| Jiang, H. et al. "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models." | EMNLP 2023, DOI `10.18653/v1/2023.emnlp-main.825` | https://aclanthology.org/2023.emnlp-main.825/ |
| Li, Y., Dong, B., Guerin, F., Lin, C. "Compressing Context to Enhance Inference Efficiency of Large Language Models." (SelectiveContext) | EMNLP 2023 | https://aclanthology.org/2023.emnlp-main.391/ |
| Jiang, H. et al. "LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression." | Accepted ACL 2024 | https://arxiv.org/abs/2310.06839 |
| Robertson, S., Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." | *Foundations and Trends in Information Retrieval*, vol. 3, no. 4, pp. 333–389 (2009), DOI `10.1561/1500000019` | https://dl.acm.org/doi/10.1561/1500000019 |

**Closest adjacent-but-distinct work found:** Laenen et al., "One-Shot Neural Network Pruning via Spectral Graph Sparsification" (ICML workshop / PMLR, https://proceedings.mlr.press/v221/laenen23a/laenen23a.pdf) applies Laplacian sparsification to *neural-network weight* graphs, not context/token graphs, and is not Physarum-based.

**Verdict:** The Kirchhoff-solve + Physarum-conductance mechanism is **not a novel algorithm** — it is 15-year-old, peer-reviewed applied-math/biology literature. No paper was found applying it to **LLM context-chunk selection**. Defensible framing: **novel application of a known dynamical-systems mechanism to a new problem domain**, not a novel algorithm.

---

## 2. Extension #1 — Dynamic flow-entropy re-pruning trigger

*Mechanism: track Shannon entropy of the per-round node-flow distribution; fire a re-prune only on genuine drift/concentration (plus a stagnation backstop), instead of a fixed schedule.*

| Paper | Venue / Year | Link |
|---|---|---|
| "EntropyCache" (entropy-of-decoded-token-distribution as KV-cache recompute trigger) | arXiv preprint only | https://arxiv.org/html/2603.18489 |
| "Entropy-Triggered Retraining as Nonequilibrium Entropy Production in Deployed ML Systems" | arXiv preprint only | https://arxiv.org/html/2601.00554 |
| Gama, J. et al. "A Survey on Concept Drift Adaptation." | *ACM Computing Surveys* (accepted-version PDF; exact volume/year not independently re-confirmed against the primary ACM page) | https://mpechen.win.tue.nl/publications/pubs/Gama_ACMCS_AdaptationCD_accepted.pdf |

**Verdict: novel combination.** Entropy-drift-triggering exists in general ML/streaming contexts (cache invalidation, concept-drift-triggered retraining), but both closest analogs are arXiv preprints, not peer-reviewed. No exact match for "Shannon entropy of a *Physarum flow-network's* per-round node-flow distribution as a re-pruning trigger" was found.

---

## 3. Extension #2 — Hybrid keep-verbatim / summarize / drop tiering

*Mechanism: medium-flow survivors get extractively summarized rather than dropped outright by a hard top-K cutoff, trading compression ratio for recall.*

| Paper | Venue / Year | Link |
|---|---|---|
| Jiang, H. et al. "LongLLMLingua." | Accepted ACL 2024 | https://arxiv.org/abs/2310.06839 |
| "Characterizing Prompt Compression Methods for Long Context RAG" | **ICML 2024** (peer-reviewed, top-tier) | https://arxiv.org/html/2407.08892v1 |
| Pan, Z. et al. "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression." | arXiv (peer-reviewed venue not confirmed) | https://arxiv.org/html/2403.12968v2 |

**Verdict: novel combination — weakens a "fully novel" claim.** "Characterizing Prompt Compression Methods for Long Context RAG" (confirmed ICML 2024) directly compares extractive vs. abstractive vs. token-pruning strategies as distinct tiers — this is the closest real match and is top-tier peer-reviewed. The genuinely new element in FCNP's version is that the 3-way tier assignment is **gated by physical-network flow value**, not by an independent relevance score; that specific gating has no found precedent, but the general differential-treatment-by-tier pattern is established.

---

## 4. Extension #3 — Persistent high-flow memory tier (promotion/demotion)

*Mechanism: elements promoted into a small cross-round persistent set after repeatedly earning high flow; demoted if they stop earning it (bounded-size memory).*

| Paper | Venue / Year | Link |
|---|---|---|
| Packer, C. et al. "MemGPT: Towards LLMs as Operating Systems." | arXiv preprint (widely cited; peer-reviewed conference acceptance not confirmed) | https://arxiv.org/abs/2310.08560 |
| Einziger, G. et al. "TinyLFU: A Highly Efficient Cache Admission Policy." | arXiv (exact peer-reviewed venue not re-confirmed this pass) | https://arxiv.org/pdf/1512.00727 |
| Yang, J. et al. "FIFO can be Better than LRU: the Power of Lazy Promotion and Quick Demotion." | **HotOS 2023 (ACM SIGOPS, peer-reviewed)**, DOI `10.1145/3593856.3595887` | https://dl.acm.org/doi/10.1145/3593856.3595887 |

**Verdict: novel combination only — substantially reduces mechanism novelty.** Bounded persistent tiers with promotion/demotion by recurring value are well-established in both LLM-agent memory literature (MemGPT's core/recall/archival tiers) and classical caching theory (ACM HotOS 2023's lazy-promotion/quick-demotion result is the strongest peer-reviewed match for the *mechanism*, though not LLM-specific). The genuinely new element is coupling promotion/demotion specifically to **flow value derived from a Kirchhoff/Physarum network** — no precedent found for that coupling.

---

## 5. Extension #4 — Compute-cost vs. token-cost quantification

*Mechanism: quantify a deterministic linear-solve's wall-clock compute cost against the metered per-call token cost of LLM-based compression, as a formal evaluation axis alongside compression ratio and F1/recall.*

No peer-reviewed academic paper was found establishing "deterministic-algorithm compute cost vs. LLM-API token cost" as a formal evaluation dimension. Only non-peer-reviewed industry/blog sources were found (not cited here — they do not meet the citation bar for a thesis). One tangential preprint:

| Paper | Venue / Year | Link |
|---|---|---|
| "A Cost-Benefit Analysis of Replacing OpenAI's LLM with Open-Source SLMs" | arXiv preprint only | https://arxiv.org/html/2312.14972v3 |

This preprint benchmarks $/token cost between SLMs and proprietary APIs — a related but different axis (model-vs-model cost, not deterministic-algorithm-vs-LLM-API cost).

**Verdict: no peer-reviewed prior art found — treat as the least-grounded but potentially most novel extension.** Absence of evidence is not evidence of absence: a further targeted search across systems venues specifically (MLSys, OSDI, SOSP) was not completed and could still surface a match. **Do one more targeted search there before asserting "first to formalize" in the defense.**

---

## 6. Extension #5 — Agricultural domain application (data.gov.in mandi prices)

*Mechanism: apply LLM context compression to a real-time open-government agricultural commodity-price feed (India's data.gov.in mandi/APMC price API), compressing 50–100+ record API responses to top-K relevant records for a farmer's natural-language query.*

| Source | Venue / Year | Link |
|---|---|---|
| "Krishi Mitra: A Multilingual AI-Powered Conversational Agent for Indian Farmers..." | **IJARCCE, May 2026 — lower-tier journal, not top-tier** | https://ijarcce.com/wp-content/uploads/2026/05/IJARCCE.2026.155127-Krishi.pdf |
| "Current Daily Price of Various Commodities from Various Markets (Mandi)" | Official data.gov.in resource, Ministry of Agriculture and Farmers Welfare (Govt. of India) — confirms the domain data source is real, not an academic citation | https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070 |

**Verdict: novel combination/application.** No top-tier peer-reviewed paper applies LLM context compression — flow-based or otherwise — specifically to mandi/APMC price API responses. Krishi Mitra (IJARCCE 2026) is the closest academic match but is (a) a lower-tier venue and (b) uses generic RAG (ChromaDB), not graph/flow-based compression. Lean on the *methodological* novelty (flow-based compression) in the defense, not on the domain itself being unclaimed — related agri-AI assistants already exist, just not with this compression method.

---

## 7. Summary table (for slide use)

| Contribution | Closest prior art | Verdict |
|---|---|---|
| Core FCNP (Kirchhoff+Physarum → LLM context pruning) | Tero et al. *Science* 2010; Bonifaci et al. *J. Theor. Biol.* 2012; baselines LLMLingua/SelectiveContext (EMNLP 2023) | **Novel application** |
| Ext. #1 flow-entropy re-pruning trigger | EntropyCache, entropy-triggered retraining (arXiv preprints only) | **Novel combination** |
| Ext. #2 keep/summarize/drop hybrid tiering | LongLLMLingua (ACL 2024); "Characterizing Prompt Compression Methods for Long Context RAG" (ICML 2024) | **Novel combination** (weaker claim) |
| Ext. #3 persistent high-flow memory tier | MemGPT (arXiv); "FIFO can be Better than LRU" (ACM HotOS 2023) | **Novel combination only** (mechanism itself not new) |
| Ext. #4 compute-cost vs. token-cost axis | No peer-reviewed match found | **Fully novel by absence of prior art** (verify further before claiming "first") |
| Ext. #5 agri data.gov.in domain application | Krishi Mitra (IJARCCE 2026, lower-tier, generic RAG) | **Novel combination/application** |

## 8. Citation corrections applied to this repository

- **Fixed:** the Bonifaci et al. 2012 DOI was previously mis-cited in `README.md` as `10.1016/j.jtbi.2011.10.021` — that DOI belongs to an unrelated paper ("Prediction of protein-protein interaction sites using patch-based residue characterization," *J. Theor. Biol.* 2012, 293:143-50). Corrected to the real DOI: `10.1016/j.jtbi.2012.06.017`.
- **Flagged, not removed:** `arXiv:2601.07190` ("Active Context Compression: Autonomous Memory Management in LLM Agents") is confirmed to genuinely exist (submitted 12 Jan 2026, author Nikhil Verma) but is **preprint-only as of Aug 2026, with no peer-reviewed acceptance found**. It is retained in `README.md` only as *motivating inspiration* for the five extensions, with an explicit preprint caveat added — do not present it as a peer-reviewed source in the defense.

## 9a. Additional verified peer-reviewed JOURNAL papers (not conference/arXiv), by topic

The citations above lean heavily on top-tier **conferences** (EMNLP, ACL, ICML, HotOS) because that is where most LLM-compression work is published. The list below adds genuine peer-reviewed **journal** articles — each independently confirmed by fetching the live page and reading its journal name, volume, and DOI off the page itself, not assumed from a search snippet.

**Core mechanism — Physarum/network-flow algorithms:**

| Paper | Journal (real, verified) | Link |
|---|---|---|
| Gharehchopogh, F.S. et al. "Slime Mould Algorithm: A Comprehensive Survey of Its Variants and Applications." | **Archives of Computational Methods in Engineering**, vol. 30, pp. 2683–2723 (2023) — Springer, high-impact review journal | https://pmc.ncbi.nlm.nih.gov/articles/PMC9838547/ |
| "Advances in Slime Mould Algorithm: A Comprehensive Survey." | **Biomimetics**, vol. 9, no. 1, 31 (2024) — MDPI, mid-tier open access | https://www.mdpi.com/2313-7673/9/1/31 |
| "An Improved Physarum polycephalum Algorithm for the Shortest Path Problem." | **The Scientific World Journal** (2014), Hindawi/PMC-indexed — mid-tier, note Hindawi's editorial reputation has been debated in recent years, cite cautiously | https://pmc.ncbi.nlm.nih.gov/articles/PMC3984829/ |

These are review/survey papers, useful for a defense to show the Physarum-algorithm-family is an active, still-growing peer-reviewed research area beyond the original 2010/2012 papers — not to replace Tero 2010 / Bonifaci 2012 as the primary mechanism citation.

**Foundational math — Laplacian / resistor-network theory underlying FCNP's Kirchhoff formulation:**

| Paper | Journal | Link |
|---|---|---|
| "Effective resistance is more than distance: Laplacians, Simplices and the Schur complement." | **Linear Algebra and its Applications** (2022), Elsevier | https://arxiv.org/pdf/2010.04521.pdf (journal DOI 10.1016/j.laa.2022.01.002) |
| "Recursion-Transform method to a non-regular m×n cobweb with an arbitrary longitude." | **Scientific Reports**, vol. 5, 11266 (2015), Nature Portfolio | https://pmc.ncbi.nlm.nih.gov/articles/PMC4466885/ |
| "Comparison of methods to determine point-to-point resistance in nearly rectangular networks with application to a 'hammock' network." | **Royal Society Open Science** (2015) | https://pmc.ncbi.nlm.nih.gov/articles/PMC4448860/ |

**Extension #3 (persistent memory / promotion-demotion) — additional journal-tier caching literature:**

| Paper | Journal | Link |
|---|---|---|
| Pires, S., Ziviani, A., Sampaio, L.N. "Contextual dimensions for cache replacement schemes in information-centric networks: a systematic review." | **PeerJ Computer Science** (2021) | https://peerj.com/articles/cs-418 |

**Extension #4 (compute-cost vs. token-cost) — revised finding: a genuine peer-reviewed journal match exists, though not an exact framing match:**

| Paper | Journal | Link |
|---|---|---|
| Klang, E. et al. "A strategy for cost-effective large language model use at health system-scale." | **npj Digital Medicine**, vol. 7, article 320 (2024) — Nature Portfolio | https://pmc.ncbi.nlm.nih.gov/articles/PMC11574261/ |

**Revised verdict for Ext. #4:** the original claim of "no peer-reviewed match found anywhere" should be softened. This npj Digital Medicine paper is a real, peer-reviewed, Nature-portfolio journal article that formally analyzes LLM cost-effectiveness at production scale — it is the closest peer-reviewed journal match found to date. It still does **not** frame the comparison the way FCNP does (deterministic linear-solve compute cost vs. metered LLM-API token cost as a formal dual axis); it instead analyzes query-batching/concatenation strategies to reduce token spend within an all-LLM pipeline. The novel element in FCNP — quantifying a *non-LLM, deterministic algorithm's* compute cost against an *LLM-API's* token cost as two sides of the same evaluation table — still has no exact peer-reviewed precedent, but the topic area itself is now shown to have real journal-tier engagement, which is a stronger position for a defense than "nothing exists."

**Extension #5 (agricultural domain) — stronger peer-reviewed comparators than IJARCCE:**

| Paper | Journal | Link |
|---|---|---|
| Ibrahim, A., Senthilkumar, K., Saito, K. "Evaluating responses by ChatGPT to farmers' questions on irrigated lowland rice cultivation in Nigeria." | **Scientific Reports**, vol. 14 (2024) — Nature Portfolio | https://pmc.ncbi.nlm.nih.gov/articles/PMC10858882/ |
| "Large language models can help boost food production, but be mindful of their risks." | **Frontiers in Artificial Intelligence** (2024) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11543567/ |
| "A Systematic Review of IoT Solutions for Smart Farming." | **Sensors** (MDPI), vol. 20, no. 15, 4231 (2020) | https://pmc.ncbi.nlm.nih.gov/articles/PMC7436012/ |

**Revised verdict for Ext. #5:** Scientific Reports (Nature Portfolio) is a substantially stronger peer-reviewed comparator than IJARCCE (lower-tier). It confirms LLM-assisted farmer Q&A is an active, top-journal-adjacent research area — but it evaluates a generic conversational LLM (ChatGPT) on rice cultivation Q&A, not a flow/graph-based context-compression method on structured price-record retrieval. The FCNP-specific angle (Kirchhoff-flow compression applied to mandi/APMC price API responses) still has no direct peer-reviewed precedent found; lean on this in the defense as the specific novel intersection, not on "no one has used LLMs for farmers" (they have).

## 9. Open items before finalizing the defense

1. Do one more targeted search in systems venues (MLSys, OSDI, SOSP) for Extension #4 before claiming no prior art exists.
2. Do not cite Vorburger & Bernstein (2006) on concept drift directly — only a secondary reference to it was found (inside the Gama et al. survey); cite Gama et al. instead, or omit.
3. MemGPT's peer-reviewed conference status beyond arXiv was not independently confirmed — treat it as arXiv-tier prior art, not a peer-reviewed one.
