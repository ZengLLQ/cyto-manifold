"""Leiden and Louvain community detection and cluster quality evaluation.

Implements pure-Python/NumPy graph clustering algorithms without external
igraph or leidenalg C++ dependencies:

1. Exact weighted Modularity (Newman-Girvan with resolution gamma) and
   Constant Potts Model (CPM) quality functions.
2. Fast greedy local node moving phase.
3. Leiden refinement phase (Traag, Waltman, & van Eck, 2019) guaranteeing that
   every refined sub-community is internally well-connected before graph
   coarsening, eliminating the disconnected-cluster pathology of Louvain.
4. Cluster validation metrics: Silhouette score, Calinski-Harabasz index,
   Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and
   induced subgraph connectivity verification.
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.types import ClusterResult, FloatArray, IntArray


def compute_modularity(
    adj: FloatArray,
    labels: IntArray,
    resolution: float = 1.0,
) -> float:
    """Compute weighted Newman-Girvan modularity with resolution ``gamma``."""
    w = np.maximum(adj, 0.0)
    k = np.sum(w, axis=1)
    two_m = float(np.sum(k))
    if two_m <= 1e-12:
        return 0.0

    q_val = 0.0
    for c in np.unique(labels):
        mask = labels == c
        e_cc = float(np.sum(w[np.ix_(mask, mask)]))
        a_c = float(np.sum(k[mask]))
        q_val += (e_cc / two_m) - resolution * ((a_c / two_m) ** 2)
    return float(q_val)


def verify_clusters_connected(adj: FloatArray, labels: IntArray, tol: float = 1e-12) -> bool:
    """Verify that every cluster induces a connected subgraph in ``adj``."""
    for c in np.unique(labels):
        nodes = np.flatnonzero(labels == c)
        if len(nodes) <= 1:
            continue
        sub = adj[np.ix_(nodes, nodes)]
        visited = {0}
        queue: deque[int] = deque([0])
        while queue:
            cur = queue.popleft()
            nbrs = np.flatnonzero(sub[cur] > tol)
            for nb in nbrs:
                nb_i = int(nb)
                if nb_i not in visited:
                    visited.add(nb_i)
                    queue.append(nb_i)
        if len(visited) != len(nodes):
            return False
    return True


def _connected_components_subset(adj: FloatArray, nodes: NDArray[np.int64], tol: float = 1e-12) -> List[List[int]]:
    """Return connected components within a node subset."""
    n_sub = len(nodes)
    if n_sub == 0:
        return []
    sub = adj[np.ix_(nodes, nodes)]
    visited = np.zeros(n_sub, dtype=bool)
    components: List[List[int]] = []
    for start in range(n_sub):
        if visited[start]:
            continue
        comp: List[int] = []
        queue: deque[int] = deque([start])
        visited[start] = True
        while queue:
            u = queue.popleft()
            comp.append(int(nodes[u]))
            for v in np.flatnonzero(sub[u] > tol):
                vi = int(v)
                if not visited[vi]:
                    visited[vi] = True
                    queue.append(vi)
        components.append(comp)
    return components


def _local_move_phase(
    adj: FloatArray,
    node_weights: FloatArray,
    init_labels: IntArray,
    resolution: float,
    two_m: float,
    rng: np.random.Generator,
    max_sweeps: int = 12,
) -> IntArray:
    """Fast local moving optimizing weighted modularity."""
    n = adj.shape[0]
    labels = init_labels.copy()
    k = np.sum(adj, axis=1)
    # Sum of degrees in each community
    max_label = int(np.max(labels)) + 1
    comm_tot = np.bincount(labels, weights=k, minlength=max_label).astype(np.float64)

    nbr_lists: List[IntArray] = [
        np.array([j for j in np.flatnonzero(adj[i] > 1e-12) if j != i], dtype=np.int64)
        for i in range(n)
    ]

    for _ in range(max_sweeps):
        moved = False
        order = rng.permutation(n)
        for u in order:
            u_i = int(u)
            nbrs = nbr_lists[u_i]
            if len(nbrs) == 0:
                continue
            cur_c = int(labels[u_i])
            k_u = float(k[u_i])

            # Remove u from current community temporarily
            comm_tot[cur_c] -= k_u

            # Aggregate edge weights from u to neighboring communities
            nbr_comms = labels[nbrs]
            nbr_weights = adj[u_i, nbrs]
            unique_comms = np.unique(nbr_comms)

            best_c = cur_c
            best_gain = -1e18

            # Evaluate modularity gain for staying in cur_c or moving to neighbor comm
            candidates = set(int(c) for c in unique_comms)
            candidates.add(cur_c)
            for cand_c in candidates:
                w_in = float(np.sum(nbr_weights[nbr_comms == cand_c]))
                tot_c = float(comm_tot[cand_c])
                gain = w_in - resolution * (k_u * tot_c / two_m)
                if gain > best_gain + 1e-12 or (abs(gain - best_gain) <= 1e-12 and cand_c == cur_c):
                    best_gain = gain
                    best_c = cand_c

            comm_tot[best_c] += k_u
            if best_c != cur_c:
                labels[u_i] = best_c
                moved = True
        if not moved:
            break

    return _relabel_contiguous(labels)


def _refine_partition(
    adj: FloatArray,
    coarse_labels: IntArray,
    resolution: float,
    two_m: float,
    rng: np.random.Generator,
    theta_rand: float = 0.01,
) -> IntArray:
    """Leiden refinement: merge singletons within each coarse community while preserving connectivity."""
    n = adj.shape[0]
    k = np.sum(adj, axis=1)
    refined = np.arange(n, dtype=np.int64)
    comm_tot = k.copy()

    for c in np.unique(coarse_labels):
        nodes = np.flatnonzero(coarse_labels == c)
        if len(nodes) <= 1:
            continue
        sub_adj = adj[np.ix_(nodes, nodes)]
        total_c_deg = float(np.sum(k[nodes]))

        # Identify well-connected nodes within community c
        for idx_u in rng.permutation(len(nodes)):
            u = int(nodes[idx_u])
            if np.sum(refined == refined[u]) > 1:
                continue  # already merged
            # Candidate sub-communities inside c that u connects to
            nbrs_local = np.flatnonzero(sub_adj[idx_u] > 1e-12)
            if len(nbrs_local) == 0:
                continue

            nbr_global = nodes[nbrs_local]
            cand_comms = np.unique(refined[nbr_global])
            best_c = int(refined[u])
            best_gain = 0.0
            scored_cands: List[Tuple[int, float]] = []

            for cand in cand_comms:
                cand_i = int(cand)
                if cand_i == best_c:
                    continue
                # Check well-connectedness criterion of target sub-community inside c
                cand_nodes = np.flatnonzero(refined == cand_i)
                ext_c = float(np.sum(adj[np.ix_(cand_nodes, nodes)])) - float(
                    np.sum(adj[np.ix_(cand_nodes, cand_nodes)])
                )
                tot_cand = float(comm_tot[cand_i])
                well_conn_thresh = resolution * tot_cand * max(total_c_deg - tot_cand, 0.0) / two_m
                if ext_c + 1e-12 < well_conn_thresh and len(cand_nodes) > 1:
                    continue

                w_in = float(np.sum(adj[u, cand_nodes]))
                gain = w_in - resolution * (float(k[u]) * tot_cand / two_m)
                if gain > 0.0:
                    scored_cands.append((cand_i, gain))
                    if gain > best_gain:
                        best_gain = gain
                        best_c = cand_i

            if scored_cands:
                # Log-sum-exp stabilized randomized positive-gain selection
                logits = np.array([g / max(theta_rand, 1e-4) for _, g in scored_cands], dtype=np.float64)
                logits -= float(np.max(logits))
                probs = np.exp(logits)
                probs /= float(np.sum(probs))
                chosen_idx = int(rng.choice(len(scored_cands), p=probs))
                target_c = scored_cands[chosen_idx][0]
                old_c = int(refined[u])
                comm_tot[old_c] -= float(k[u])
                comm_tot[target_c] += float(k[u])
                refined[u] = target_c

    # Enforce strict connectivity inside every refined cluster
    final_refined = np.zeros(n, dtype=np.int64)
    next_id = 0
    for r_c in np.unique(refined):
        r_nodes = np.flatnonzero(refined == r_c)
        for comp in _connected_components_subset(adj, r_nodes):
            for node_idx in comp:
                final_refined[node_idx] = next_id
            next_id += 1
    return final_refined


def _coarsen_graph(adj: FloatArray, partition: IntArray) -> FloatArray:
    """Aggregate weighted adjacency matrix onto refined super-nodes."""
    n_comm = int(np.max(partition)) + 1
    p_mat = np.zeros((adj.shape[0], n_comm), dtype=np.float64)
    p_mat[np.arange(adj.shape[0]), partition] = 1.0
    coarse = p_mat.T @ adj @ p_mat
    return 0.5 * (coarse + coarse.T)


def _relabel_contiguous(labels: IntArray) -> IntArray:
    """Map arbitrary integer labels to contiguous 0..K-1 sorted by size descending."""
    unique, counts = np.unique(labels, return_counts=True)
    order = unique[np.argsort(-counts, kind="mergesort")]
    mapping = {int(old): new for new, old in enumerate(order)}
    return np.array([mapping[int(x)] for x in labels], dtype=np.int64)


def leiden_cluster(
    adj: FloatArray,
    resolution: float = 1.0,
    n_iterations: int = 4,
    seed: int = 42,
) -> IntArray:
    """Run multi-level Leiden community detection on a symmetric weighted graph.

    Parameters
    ----------
    adj:
        Symmetric non-negative adjacency/connectivity matrix ``(n_cells, n_cells)``.
    resolution:
        Modularity resolution parameter ``gamma`` (larger values yield finer clusters).
    n_iterations:
        Number of multi-level Leiden restarts/refinement passes.
    seed:
        Random seed controlling node visit permutation.
    """
    n = adj.shape[0]
    w_base = 0.5 * (np.maximum(adj, 0.0) + np.maximum(adj, 0.0).T)
    np.fill_diagonal(w_base, 0.0)
    two_m = float(np.sum(w_base))
    if two_m <= 1e-12:
        return np.zeros(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    best_labels = np.arange(n, dtype=np.int64)
    best_q = -np.inf

    for _ in range(max(1, n_iterations)):
        cur_labels = np.arange(n, dtype=np.int64)
        cur_adj = w_base.copy()
        node_to_super = np.arange(n, dtype=np.int64)

        for _level in range(10):
            n_cur = cur_adj.shape[0]
            if n_cur <= 1:
                break
            init_part = np.arange(n_cur, dtype=np.int64)
            moved = _local_move_phase(
                cur_adj,
                np.ones(n_cur, dtype=np.float64),
                init_part,
                resolution=resolution,
                two_m=two_m,
                rng=rng,
            )
            if int(np.max(moved)) + 1 == n_cur:
                break

            refined = _refine_partition(
                cur_adj, moved, resolution=resolution, two_m=two_m, rng=rng
            )

            # Map original nodes through refined super-nodes
            node_to_super = refined[node_to_super]

            # Build aggregate coarse graph retaining coarse-level initial grouping
            cur_adj = _coarsen_graph(cur_adj, refined)
            cur_labels = _relabel_contiguous(node_to_super)
            if int(np.max(refined)) + 1 == n_cur:
                break

        # Final connected-component enforcement on original graph
        enforced = np.zeros(n, dtype=np.int64)
        next_id = 0
        for c in np.unique(cur_labels):
            c_nodes = np.flatnonzero(cur_labels == c)
            for comp in _connected_components_subset(w_base, c_nodes):
                for idx in comp:
                    enforced[idx] = next_id
                next_id += 1
        enforced = _relabel_contiguous(enforced)
        q_cur = compute_modularity(w_base, enforced, resolution=resolution)
        if q_cur > best_q:
            best_q = q_cur
            best_labels = enforced

    return best_labels


def louvain_cluster(
    adj: FloatArray,
    resolution: float = 1.0,
    seed: int = 42,
) -> IntArray:
    """Run classic multi-level Louvain community detection for comparison."""
    n = adj.shape[0]
    w_base = 0.5 * (np.maximum(adj, 0.0) + np.maximum(adj, 0.0).T)
    np.fill_diagonal(w_base, 0.0)
    two_m = float(np.sum(w_base))
    if two_m <= 1e-12:
        return np.zeros(n, dtype=np.int64)

    rng = np.random.default_rng(seed)
    cur_adj = w_base.copy()
    node_to_super = np.arange(n, dtype=np.int64)
    for _ in range(10):
        n_cur = cur_adj.shape[0]
        moved = _local_move_phase(
            cur_adj,
            np.ones(n_cur, dtype=np.float64),
            np.arange(n_cur, dtype=np.int64),
            resolution=resolution,
            two_m=two_m,
            rng=rng,
        )
        node_to_super = moved[node_to_super]
        if int(np.max(moved)) + 1 == n_cur:
            break
        cur_adj = _coarsen_graph(cur_adj, moved)
    return _relabel_contiguous(node_to_super)


def compute_silhouette_score(features: FloatArray, labels: IntArray) -> float:
    """Compute mean Silhouette Coefficient in [-1, 1] over feature embeddings."""
    unique = np.unique(labels)
    if len(unique) < 2 or features.shape[0] <= len(unique):
        return 0.0
    diff = features[:, None, :] - features[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))

    scores = np.zeros(features.shape[0], dtype=np.float64)
    for i in range(features.shape[0]):
        c_i = labels[i]
        same = (labels == c_i) & (np.arange(features.shape[0]) != i)
        a_i = float(np.mean(dist[i, same])) if np.any(same) else 0.0
        b_i = np.inf
        for c_other in unique:
            if c_other == c_i:
                continue
            other_mask = labels == c_other
            b_i = min(b_i, float(np.mean(dist[i, other_mask])))
        denom = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0
    return float(np.mean(scores))


def compute_calinski_harabasz(features: FloatArray, labels: IntArray) -> float:
    """Compute Calinski-Harabasz variance ratio criterion."""
    unique = np.unique(labels)
    k = len(unique)
    n = features.shape[0]
    if k < 2 or n <= k:
        return 0.0
    overall_mean = np.mean(features, axis=0)
    bg_disp = 0.0
    wg_disp = 0.0
    for c in unique:
        pts = features[labels == c]
        c_mean = np.mean(pts, axis=0)
        bg_disp += len(pts) * float(np.sum((c_mean - overall_mean) ** 2))
        wg_disp += float(np.sum((pts - c_mean[None, :]) ** 2))
    if wg_disp <= 1e-12:
        return 0.0
    return float((bg_disp / (k - 1)) / (wg_disp / (n - k)))


def adjusted_rand_index(labels_true: Sequence[int] | IntArray, labels_pred: Sequence[int] | IntArray) -> float:
    """Compute Adjusted Rand Index (ARI) between two clusterings."""
    t = np.asarray(labels_true, dtype=np.int64)
    p = np.asarray(labels_pred, dtype=np.int64)
    if t.shape != p.shape or len(t) == 0:
        raise ValueError("Label arrays must have identical non-empty shape")

    _, t_idx = np.unique(t, return_inverse=True)
    _, p_idx = np.unique(p, return_inverse=True)
    n_t = int(np.max(t_idx)) + 1
    n_p = int(np.max(p_idx)) + 1

    contingency = np.zeros((n_t, n_p), dtype=np.int64)
    np.add.at(contingency, (t_idx, p_idx), 1)

    sum_comb_c = float(np.sum(comb(contingency, 2)))
    sum_comb_a = float(np.sum(comb(np.sum(contingency, axis=1), 2)))
    sum_comb_b = float(np.sum(comb(np.sum(contingency, axis=0), 2)))
    total_comb = float(comb(len(t), 2))
    if total_comb <= 0.0:
        return 1.0

    expected = (sum_comb_a * sum_comb_b) / total_comb
    max_idx = 0.5 * (sum_comb_a + sum_comb_b)
    denom = max_idx - expected
    if abs(denom) <= 1e-12:
        return 1.0 if sum_comb_c == expected else 0.0
    return float((sum_comb_c - expected) / denom)


def normalized_mutual_information(
    labels_true: Sequence[int] | IntArray,
    labels_pred: Sequence[int] | IntArray,
) -> float:
    """Compute arithmetic-mean Normalized Mutual Information (NMI) in [0, 1]."""
    t = np.asarray(labels_true, dtype=np.int64)
    p = np.asarray(labels_pred, dtype=np.int64)
    n = float(len(t))
    if n == 0:
        return 0.0

    _, t_idx = np.unique(t, return_inverse=True)
    _, p_idx = np.unique(p, return_inverse=True)
    n_t = int(np.max(t_idx)) + 1
    n_p = int(np.max(p_idx)) + 1

    contingency = np.zeros((n_t, n_p), dtype=np.float64)
    np.add.at(contingency, (t_idx, p_idx), 1.0)
    p_ij = contingency / n
    p_i = np.sum(p_ij, axis=1, keepdims=True)
    p_j = np.sum(p_ij, axis=0, keepdims=True)

    nz = p_ij > 0.0
    mi = float(np.sum(p_ij[nz] * np.log(p_ij[nz] / (p_i @ p_j)[nz])))
    h_t = -float(np.sum(p_i[p_i > 0.0] * np.log(p_i[p_i > 0.0])))
    h_p = -float(np.sum(p_j[p_j > 0.0] * np.log(p_j[p_j > 0.0])))
    denom = 0.5 * (h_t + h_p)
    if denom <= 1e-12:
        return 1.0
    return float(np.clip(mi / denom, 0.0, 1.0))


def cluster_dataset(
    dataset: TranscriptomeDataset,
    resolution: float = 1.0,
    method: str = "leiden",
    seed: int = 42,
    obs_key: Optional[str] = None,
) -> ClusterResult:
    """Cluster cells using the stored ``connectivities`` graph and save labels in ``obs``."""
    if "connectivities" not in dataset.obsp:
        raise ValueError("dataset.obsp['connectivities'] is required before clustering")
    adj = dataset.obsp["connectivities"]

    if method == "leiden":
        labels = leiden_cluster(adj, resolution=resolution, seed=seed)
    elif method == "louvain":
        labels = louvain_cluster(adj, resolution=resolution, seed=seed)
    else:
        raise ValueError(f"Unsupported clustering method '{method}'; expected 'leiden' or 'louvain'")

    key = obs_key or method
    dataset.obs[key] = labels
    dataset.obs["cluster"] = labels

    feats = dataset.obsm["X_pca"] if "X_pca" in dataset.obsm else dataset.X
    mod = compute_modularity(adj, labels, resolution=resolution)
    sil = compute_silhouette_score(feats, labels)
    ch = compute_calinski_harabasz(feats, labels)
    conn = verify_clusters_connected(adj, labels)
    unique, counts = np.unique(labels, return_counts=True)
    sizes = {int(c): int(cnt) for c, cnt in zip(unique, counts)}

    return ClusterResult(
        labels=labels,
        n_clusters=len(unique),
        modularity=mod,
        silhouette=sil,
        calinski_harabasz=ch,
        is_connected=conn,
        resolution=resolution,
        method=method,
        cluster_sizes=sizes,
    )
