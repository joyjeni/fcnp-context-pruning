"""Flow-Based Context Network Pruning (FCNP).

Models the context store as an undirected weighted graph G = (V, E):

    - V : context elements
    - E : semantic similarity edges (cosine of embeddings, thresholded)

Each edge has a conductance D_ij(t) updated by iterative current
reinforcement on a resistor network. Nodes act as current sources
(query-relevant elements) or sink (virtual task node).

Update rule
-----------
At each iteration t:

    1. Solve Kirchhoff's current law on the weighted graph:

           sum_j D_ij * (p_i - p_j) = I_i

       where I_i is injected current at node i.

    2. Edge currents:

           Q_ij = D_ij * (p_i - p_j)

    3. Non-linear conductance reinforcement:

           D_ij(t+1) = (1 - mu) * D_ij(t) + alpha * |Q_ij|^gamma

    4. Stop when:

           sum_{ij} |D_ij(t+1) - D_ij(t)| < epsilon * sum_{ij} D_ij(t)

    5. Score nodes by aggregate incident flow; keep top-K.

References
----------
- Tero, A. et al. "Rules for biologically inspired adaptive network
  design." Science 327 (2010).
- Bonifaci, V., Mehlhorn, K., Varma, G. "Physarum can compute shortest
  paths." J. Theor. Biol. 309 (2012).
- Spielman, D. A. "Spectral graph theory and its applications." FOCS
  (2007).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fcnp.types import ContextElement, PruneResult


@dataclass
class FCNPConfig:
    similarity_threshold: float = 0.30   # min cosine to create an edge
    epsilon: float = 1e-4                # convergence threshold (relative)
    max_iterations: int = 200
    mu: float = 0.10                     # decay rate of conductance
    alpha: float = 0.50                  # reinforcement gain
    gamma: float = 1.20                  # non-linearity exponent (>1)
    keep_top_k_fraction: float = 0.10    # node retention budget after prune
    current_injection: float = 1.0
    laplacian_regularization: float = 1e-9


class FlowBasedNetworkPruner:
    """Iterative current-reinforced pruning of a semantic context graph."""

    def __init__(self, config: FCNPConfig | None = None):
        self.config = config or FCNPConfig()
        self.iterations_ran: int = 0
        self.converged: bool = False

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def _build_adjacency(self, elements: list[ContextElement]) -> np.ndarray:
        """Thresholded cosine-similarity adjacency."""
        n = len(elements)
        if n == 0:
            return np.zeros((0, 0))

        embeddings = np.stack([e.embedding for e in elements]).astype(np.float64)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms

        sim = normed @ normed.T
        np.fill_diagonal(sim, 0.0)
        sim = np.where(sim < self.config.similarity_threshold, 0.0, sim)
        # Clip negatives (cosine in [-1,1]) — only positive similarity creates edges
        sim = np.clip(sim, 0.0, None)
        return sim

    def _current_injection(
        self,
        elements: list[ContextElement],
        query_embedding: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute source mass per element and the sink coupling vector.

        Returns
        -------
        I : np.ndarray, shape (n+1,)
            Injected current per node; sink is node index n with
            current = -sum(sources).
        sink_coupling : np.ndarray, shape (n,)
            Mass-weighted conductance from each element to the virtual
            sink node.
        """
        n = len(elements)
        I = np.zeros(n + 1)

        if query_embedding is None:
            mass = np.array([e.importance for e in elements], dtype=np.float64)
        else:
            embeddings = np.stack([e.embedding for e in elements]).astype(np.float64)
            q = query_embedding.astype(np.float64)
            q = q / (np.linalg.norm(q) + 1e-12)
            embn = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
            mass = np.clip(embn @ q, 0.0, None)

        if mass.sum() == 0:
            mass = np.ones(n) / n
        else:
            mass = mass / mass.sum()

        I[:n] = mass * self.config.current_injection
        I[n] = -self.config.current_injection  # sink absorbs total current
        return I, mass

    @staticmethod
    def _augment_with_sink(D: np.ndarray, sink_coupling: np.ndarray) -> np.ndarray:
        """Append virtual sink node connected to every element."""
        n = D.shape[0]
        aug = np.zeros((n + 1, n + 1))
        aug[:n, :n] = D
        aug[:n, n] = sink_coupling
        aug[n, :n] = sink_coupling
        return aug

    # ------------------------------------------------------------------
    # Kirchhoff solver
    # ------------------------------------------------------------------
    def _solve_potentials(self, D: np.ndarray, I: np.ndarray) -> np.ndarray:
        """Solve L p = I where L = diag(D 1) - D, grounding the sink at 0."""
        deg = D.sum(axis=1)
        L = np.diag(deg) - D

        # Ground last node (sink) by removing its row/col
        L_red = L[:-1, :-1] + self.config.laplacian_regularization * np.eye(L.shape[0] - 1)
        I_red = I[:-1]

        try:
            p_red = np.linalg.solve(L_red, I_red)
        except np.linalg.LinAlgError:
            p_red, *_ = np.linalg.lstsq(L_red, I_red, rcond=None)

        return np.concatenate([p_red, [0.0]])

    @staticmethod
    def _edge_currents(D: np.ndarray, p: np.ndarray) -> np.ndarray:
        diff = p[:, None] - p[None, :]
        return np.abs(D * diff)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def prune(
        self,
        elements: list[ContextElement],
        query_embedding: np.ndarray | None = None,
    ) -> PruneResult:
        """Run current-reinforcement loop and return pruned elements.

        Parameters
        ----------
        elements : list[ContextElement]
        query_embedding : np.ndarray | None
            Optional dense vector for the downstream task. Elements
            aligned with the query become stronger current sources.

        Returns
        -------
        PruneResult
        """
        cfg = self.config
        n = len(elements)

        input_tokens = sum(e.token_count() for e in elements)

        if n == 0:
            self.iterations_ran = 0
            self.converged = True
            return PruneResult(
                survivors=[], n_input=0, n_output=0,
                input_tokens=0, output_tokens=0,
                iterations=0, converged=True,
                node_flow=np.zeros(0),
            )

        adjacency = self._build_adjacency(elements)
        I, sink_coupling = self._current_injection(elements, query_embedding)

        D_aug = self._augment_with_sink(adjacency, sink_coupling)

        prev_total = D_aug.sum()
        self.iterations_ran = 0
        self.converged = False

        for it in range(cfg.max_iterations):
            p = self._solve_potentials(D_aug, I)
            Q = self._edge_currents(D_aug, p)

            reinforcement = cfg.alpha * np.power(Q, cfg.gamma)
            D_new = (1.0 - cfg.mu) * D_aug + reinforcement
            np.fill_diagonal(D_new, 0.0)

            delta = np.abs(D_new - D_aug).sum()
            D_aug = D_new
            self.iterations_ran = it + 1

            if delta < cfg.epsilon * max(prev_total, 1.0):
                self.converged = True
                break
            prev_total = D_aug.sum()

        # Aggregate flow per real node (sum across all incident edges including sink)
        node_flow = D_aug[:n, :].sum(axis=1)

        # Keep top-K nodes
        keep_k = max(1, int(np.ceil(cfg.keep_top_k_fraction * n)))
        order = np.argsort(-node_flow)
        top_idx = order[:keep_k]

        survivors = [elements[i] for i in top_idx]
        max_flow = float(node_flow.max()) if node_flow.max() > 0 else 1.0
        for i, idx in enumerate(top_idx):
            survivors[i].importance = float(node_flow[idx] / max_flow)

        output_tokens = sum(e.token_count() for e in survivors)

        return PruneResult(
            survivors=survivors,
            n_input=n,
            n_output=len(survivors),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            iterations=self.iterations_ran,
            converged=self.converged,
            node_flow=node_flow,
        )
