"""Diffusion Maps, Diffusion Pseudotime (DPT), PAGA, and absorbing Markov fate probabilities.

Implements trajectory inference combining undirected manifold diffusion with
directed RNA velocity Markov transitions:

1. Anisotropic density-normalized diffusion operator (Coifman & Lafon, 2006)
   ``K_alpha(x, y) = K(x, y) / (q(x)^alpha * q(y)^alpha)`` and symmetric
   eigen-decomposition yielding diffusion components ``psi_l`` and eigenvalues
   ``lambda_l``.
2. Multiscale Diffusion Pseudotime (Haghverdi et al., 2016) accumulated over
   all non-trivial diffusion scales ``lambda_l / (1 - lambda_l)``.
3. Velocity-informed directed Markov chain ``T = (1 - eta) T_struct + eta T_vel``
   with terminal cluster/state identification and exact absorbing linear
   system solve ``(I - Q) B = R`` yielding lineage fate probabilities
   satisfying ``sum_k B_{ik} = 1`` and absorption entropy (potency).
4. Partition-based Graph Abstraction (PAGA) statistical connectivity matrix
   between Leiden clusters.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import linalg

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.types import FloatArray, IntArray, PseudotimeResult


def compute_diffusion_map(
    connectivities: FloatArray,
    n_components: int = 10,
    alpha: float = 0.5,
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Compute density-normalized diffusion map eigenvectors and Markov matrix."""
    n_cells = connectivities.shape[0]
    w = np.maximum(0.5 * (connectivities + connectivities.T), 0.0)
    np.fill_diagonal(w, 1.0)

    q = np.maximum(np.sum(w, axis=1), 1e-12)
    q_inv_alpha = np.power(q, -alpha)
    k_alpha = (q_inv_alpha[:, None] * w) * q_inv_alpha[None, :]

    d_alpha = np.maximum(np.sum(k_alpha, axis=1), 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d_alpha)
    t_sym = (d_inv_sqrt[:, None] * k_alpha) * d_inv_sqrt[None, :]
    t_sym = 0.5 * (t_sym + t_sym.T)

    # Row-stochastic Markov matrix T = D^-1 K_alpha
    t_markov = k_alpha / d_alpha[:, None]

    eigvals, eigvecs = np.linalg.eigh(t_sym)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Map symmetric eigenvectors back to right eigenvectors of T: psi = D^-1/2 phi
    psi = d_inv_sqrt[:, None] * eigvecs
    # Drop trivial stationary mode (index 0, eigenvalue ~ 1)
    k_eff = max(2, min(n_components, n_cells - 1))
    dc = psi[:, 1 : k_eff + 1].copy()
    lambdas = np.clip(eigvals[1 : k_eff + 1], -0.999, 0.999)

    for j in range(dc.shape[1]):
        norm_j = float(np.linalg.norm(dc[:, j]))
        if norm_j > 1e-12:
            dc[:, j] /= norm_j
        max_idx = int(np.argmax(np.abs(dc[:, j])))
        if dc[max_idx, j] < 0.0:
            dc[:, j] *= -1.0

    return dc, lambdas, t_markov


def compute_diffusion_pseudotime(
    diffusion_components: FloatArray,
    eigenvalues: FloatArray,
    root_cell: int,
) -> FloatArray:
    """Compute multiscale Diffusion Pseudotime (DPT) from ``root_cell``."""
    n_cells = diffusion_components.shape[0]
    if not (0 <= root_cell < n_cells):
        raise ValueError(f"root_cell={root_cell} out of bounds [0, {n_cells})")

    # Multiscale weights lambda / (1 - lambda)
    scales = np.divide(
        np.abs(eigenvalues),
        np.maximum(1.0 - np.abs(eigenvalues), 1e-4),
    )
    scaled = diffusion_components * scales[None, :]
    root_vec = scaled[root_cell]
    dists = np.linalg.norm(scaled - root_vec[None, :], axis=1)
    max_d = float(np.max(dists))
    if max_d > 1e-12:
        dists = dists / max_d
    return np.asarray(dists, dtype=np.float64)


def select_root_cell(
    dataset: TranscriptomeDataset,
    diffusion_components: FloatArray,
    velocity_transition: Optional[FloatArray] = None,
) -> int:
    """Heuristically select an early progenitor root cell.

    Prefers cells with earliest latent time (if available) or minimal
    incoming velocity transition mass, falling back to extreme first
    diffusion component pole.
    """
    if "latent_time" in dataset.obs:
        lt = np.asarray(dataset.obs["latent_time"], dtype=np.float64)
        return int(np.argmin(lt))
    if velocity_transition is not None:
        in_flux = np.sum(velocity_transition, axis=0)
        return int(np.argmin(in_flux))
    return int(np.argmin(diffusion_components[:, 0]))


def compute_paga_connectivity(
    connectivities: FloatArray,
    labels: IntArray,
) -> Tuple[FloatArray, List[int]]:
    """Compute PAGA inter-cluster statistical connectivity confidence in [0, 1]."""
    cluster_ids = [int(c) for c in np.unique(labels)]
    k = len(cluster_ids)
    w = np.maximum(0.5 * (connectivities + connectivities.T), 0.0)
    np.fill_diagonal(w, 0.0)

    n_total = float(w.shape[0])
    total_edges = float(np.sum(w > 1e-6))
    paga = np.zeros((k, k), dtype=np.float64)

    for i_idx, ci in enumerate(cluster_ids):
        mask_i = labels == ci
        n_i = float(np.sum(mask_i))
        k_i = float(np.sum(w[mask_i] > 1e-6))
        for j_idx, cj in enumerate(cluster_ids):
            if i_idx == j_idx:
                paga[i_idx, j_idx] = 1.0
                continue
            if j_idx < i_idx:
                paga[i_idx, j_idx] = paga[j_idx, i_idx]
                continue
            mask_j = labels == cj
            n_j = float(np.sum(mask_j))
            k_j = float(np.sum(w[mask_j] > 1e-6))

            e_ij = float(np.sum(w[np.ix_(mask_i, mask_j)] > 1e-6))
            expected_ij = (n_i * k_j + n_j * k_i) / max(n_total, 1.0)
            # Normalized ratio relative to random expectation and cluster size
            denom = max(min(n_i, n_j) * max(total_edges / max(n_total, 1.0), 1.0), 1.0)
            conf = max(0.0, e_ij - 0.5 * expected_ij) / denom
            paga[i_idx, j_idx] = float(np.clip(conf * 2.0, 0.0, 1.0))

    return paga, cluster_ids


def _identify_terminal_sets(
    combined_transition: FloatArray,
    labels: IntArray,
    dpt: FloatArray,
    n_terminal: Optional[int] = None,
) -> Tuple[List[int], Dict[int, IntArray]]:
    """Identify absorbing terminal macrostates via stationary flux and late pseudotime."""
    n_cells = combined_transition.shape[0]
    # Power iteration to approximate stationary distribution
    pi = np.full(n_cells, 1.0 / float(n_cells), dtype=np.float64)
    for _ in range(40):
        pi = pi @ combined_transition
        pi /= max(float(np.sum(pi)), 1e-12)

    cluster_ids = [int(c) for c in np.unique(labels)]
    if len(cluster_ids) == 1:
        c0 = cluster_ids[0]
        top_cells = np.argsort(-dpt)[: max(2, min(5, n_cells // 5))]
        return [c0], {c0: top_cells.astype(np.int64)}

    # Score clusters by stationary density and late pseudotime, avoiding low out-flow roots
    cluster_scores: List[Tuple[float, int]] = []
    for cid in cluster_ids:
        mask = labels == cid
        mean_pi = float(np.mean(pi[mask]))
        mean_dpt = float(np.mean(dpt[mask]))
        # Self-retention probability inside cluster
        sub = combined_transition[np.ix_(mask, mask)]
        retention = float(np.mean(np.sum(sub, axis=1)))
        score = 0.45 * mean_dpt + 0.35 * retention + 0.20 * (mean_pi * n_cells)
        cluster_scores.append((score, cid))

    cluster_scores.sort(reverse=True)
    k_term = n_terminal if n_terminal is not None else min(len(cluster_ids) - 1, max(2, len(cluster_ids) // 2))
    k_term = max(1, min(k_term, len(cluster_ids)))

    chosen_clusters = [cid for _, cid in cluster_scores[:k_term]]
    terminal_dict: Dict[int, IntArray] = {}
    for cid in chosen_clusters:
        c_nodes = np.flatnonzero(labels == cid)
        # Take top high-stationary / late-pseudotime core cells inside each terminal cluster
        n_core = max(2, min(len(c_nodes), max(2, len(c_nodes) // 3)))
        node_score = pi[c_nodes] * n_cells + dpt[c_nodes]
        core = c_nodes[np.argsort(-node_score, kind="mergesort")[:n_core]]
        terminal_dict[cid] = core.astype(np.int64)

    return chosen_clusters, terminal_dict


def compute_absorbing_fates(
    transition_matrix: FloatArray,
    terminal_dict: Dict[int, IntArray],
) -> Tuple[FloatArray, FloatArray]:
    """Solve exact Markov absorption probabilities ``(I - Q) B = R``.

    Transient states converge toward absorbing terminal macrostate sets.
    For every cell ``i``, ``sum_k B_{ik} == 1.0`` and ``B_{ik} in [0, 1]``.
    Also returns normalized Shannon absorption entropy (plasticity/potency).
    """
    n_cells = transition_matrix.shape[0]
    terminal_cids = sorted(terminal_dict.keys())
    k_term = len(terminal_cids)
    fate_probs = np.zeros((n_cells, k_term), dtype=np.float64)

    absorbing_mask = np.zeros(n_cells, dtype=bool)
    cell_to_term_col: Dict[int, int] = {}
    for col_idx, cid in enumerate(terminal_cids):
        for cell_idx in terminal_dict[cid]:
            ci = int(cell_idx)
            absorbing_mask[ci] = True
            cell_to_term_col[ci] = col_idx
            fate_probs[ci, col_idx] = 1.0

    transient_idx = np.flatnonzero(~absorbing_mask)
    if len(transient_idx) > 0:
        q_mat = transition_matrix[np.ix_(transient_idx, transient_idx)]
        r_mat = np.zeros((len(transient_idx), k_term), dtype=np.float64)
        for col_idx, cid in enumerate(terminal_cids):
            term_nodes = terminal_dict[cid]
            r_mat[:, col_idx] = np.sum(transition_matrix[np.ix_(transient_idx, term_nodes)], axis=1)

        # Solve (I - Q) B = R with slight diagonal regularization for near-disconnected components
        eye_q = np.eye(len(transient_idx), dtype=np.float64)
        lhs = eye_q - q_mat + 1e-7 * eye_q
        b_mat = linalg.solve(lhs, r_mat)
        b_mat = np.clip(b_mat, 0.0, 1.0)
        row_sums = np.sum(b_mat, axis=1, keepdims=True)
        valid = row_sums[:, 0] > 1e-12
        b_mat[valid] /= row_sums[valid]
        b_mat[~valid] = 1.0 / float(k_term)
        fate_probs[transient_idx] = b_mat

    # Normalized absorption entropy
    if k_term <= 1:
        entropy = np.zeros(n_cells, dtype=np.float64)
    else:
        max_ent = np.log(float(k_term))
        entropy = np.zeros(n_cells, dtype=np.float64)
        for i in range(n_cells):
            p = fate_probs[i]
            nz = p > 1e-12
            entropy[i] = -float(np.sum(p[nz] * np.log(p[nz]))) / max_ent
        entropy = np.clip(entropy, 0.0, 1.0)

    return fate_probs, entropy


def infer_trajectory_and_fates(
    dataset: TranscriptomeDataset,
    root_cell: Optional[int] = None,
    velocity_weight: float = 0.5,
    n_diffusion_components: int = 10,
    n_terminal_states: Optional[int] = None,
) -> PseudotimeResult:
    """Compute diffusion map, DPT pseudotime, PAGA graph, and absorbing lineage fates."""
    if "connectivities" not in dataset.obsp:
        raise ValueError("dataset.obsp['connectivities'] required for trajectory inference")

    conn = dataset.obsp["connectivities"]
    dc, lambdas, t_struct = compute_diffusion_map(
        conn, n_components=n_diffusion_components
    )
    t_vel = dataset.obsp.get("velocity_transition")

    if root_cell is None:
        root_cell = select_root_cell(dataset, dc, t_vel)

    dpt = compute_diffusion_pseudotime(dc, lambdas, root_cell=root_cell)

    if t_vel is not None:
        eta = float(np.clip(velocity_weight, 0.0, 1.0))
        combined = (1.0 - eta) * t_struct + eta * t_vel
    else:
        combined = t_struct.copy()
    combined /= np.maximum(np.sum(combined, axis=1, keepdims=True), 1e-12)

    if "cluster" in dataset.obs:
        labels = np.asarray(dataset.obs["cluster"], dtype=np.int64)
    else:
        labels = np.zeros(dataset.n_cells, dtype=np.int64)

    paga_mat, paga_cids = compute_paga_connectivity(conn, labels)
    term_clusters, term_dict = _identify_terminal_sets(
        combined, labels, dpt, n_terminal=n_terminal_states
    )
    fate_probs, abs_entropy = compute_absorbing_fates(combined, term_dict)

    term_state_arr = np.full(dataset.n_cells, -1, dtype=np.int64)
    for cid, nodes in term_dict.items():
        term_state_arr[nodes] = cid

    dataset.set_obsm("X_diffmap", dc)
    dataset.set_obsm("fate_probabilities", fate_probs)
    dataset.obs["dpt_pseudotime"] = dpt
    dataset.obs["absorption_entropy"] = abs_entropy
    dataset.obs["terminal_state"] = term_state_arr

    return PseudotimeResult(
        diffusion_components=dc,
        diffusion_eigenvalues=lambdas,
        root_cell=root_cell,
        dpt_pseudotime=dpt,
        terminal_states=term_state_arr,
        terminal_clusters=sorted(term_clusters),
        fate_probabilities=fate_probs,
        absorption_entropy=abs_entropy,
        combined_transition=combined,
        paga_connectivity=paga_mat,
        paga_cluster_ids=paga_cids,
    )
