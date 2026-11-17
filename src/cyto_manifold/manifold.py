"""Manifold learning, fuzzy simplicial k-NN graphs, MAGIC smoothing, and 2D layouts.

Implements graph-based manifold geometry strictly in pure NumPy and SciPy:

1. Exact k-nearest-neighbor search (Euclidean or Cosine metric).
2. UMAP smooth local fuzzy simplicial set kernel: binary search per cell for
   local bandwidth sigma_i solving sum_j exp(-(d_ij - rho_i) / sigma_i) = log2(k),
   followed by probabilistic t-conorm symmetrization W = A + A^T - A * A^T.
3. MAGIC / Markov diffusion smoothing (van Dijk et al., 2018) to denoise
   sparse spliced/unspliced counts across manifold neighbors.
4. Normalized graph Laplacian spectral embedding.
5. 2D low-dimensional manifold layout via UMAP-style attractive/repulsive SGD
   cross-entropy optimization initialized from spectral coordinates.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.types import FloatArray, NeighborGraph


def build_knn_graph(
    features: FloatArray,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    local_connectivity: float = 1.0,
) -> NeighborGraph:
    """Construct a smooth fuzzy simplicial k-nearest-neighbor cell graph.

    Parameters
    ----------
    features:
        Cell embedding array of shape ``(n_cells, n_dims)``, typically ``X_pca``.
    n_neighbors:
        Number of nearest neighbors ``k`` per cell (automatically capped to ``n_cells - 1``).
    metric:
        Distance metric (``"euclidean"`` or ``"cosine"``).
    local_connectivity:
        UMAP local connectivity parameter controlling distance to nearest neighbor.
    """
    pts = np.asarray(features, dtype=np.float64)
    n_cells = pts.shape[0]
    if n_cells < 3:
        raise ValueError("Need at least 3 cells to build a neighbor graph")

    k = max(2, min(n_neighbors, n_cells - 1))

    if metric == "euclidean":
        tree = cKDTree(pts)
        dists, indices = tree.query(pts, k=k + 1)
        dists = dists[:, 1:]
        indices = indices[:, 1:].astype(np.int64)
    elif metric == "cosine":
        norms = np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-12)
        unit_pts = pts / norms
        dist_mat = cdist(unit_pts, unit_pts, metric="cosine")
        np.fill_diagonal(dist_mat, np.inf)
        indices = np.argsort(dist_mat, axis=1, kind="mergesort")[:, :k].astype(np.int64)
        dists = np.take_along_axis(dist_mat, indices, axis=1)
        dists = np.maximum(dists, 0.0)
    else:
        raise ValueError(f"Unsupported neighbor metric '{metric}'; expected 'euclidean' or 'cosine'")

    target = float(np.log2(k))
    sigmas = np.zeros(n_cells, dtype=np.float64)
    rhos = np.zeros(n_cells, dtype=np.float64)
    directed = np.zeros((n_cells, n_cells), dtype=np.float64)

    for i in range(n_cells):
        d_i = dists[i]
        nz = d_i[d_i > 1e-12]
        if len(nz) > 0:
            idx_rho = max(0, min(int(local_connectivity) - 1, len(nz) - 1))
            rho_i = float(nz[idx_rho])
        else:
            rho_i = 0.0
        rhos[i] = rho_i

        sigma_i = _solve_smooth_knn_sigma(d_i, rho_i, target)
        sigmas[i] = sigma_i

        weights = np.exp(-np.maximum(d_i - rho_i, 0.0) / max(sigma_i, 1e-8))
        weights[d_i <= rho_i] = 1.0
        directed[i, indices[i]] = weights

    # Fuzzy set union (probabilistic t-conorm): A + A^T - A * A^T
    sym = directed + directed.T - (directed * directed.T)
    np.fill_diagonal(sym, 0.0)
    sym = np.clip(sym, 0.0, 1.0)

    return NeighborGraph(
        n_cells=n_cells,
        n_neighbors=k,
        metric=metric,
        indices=indices,
        distances=dists,
        connectivities=sym,
        sigmas=sigmas,
        rhos=rhos,
    )


def _solve_smooth_knn_sigma(
    distances: FloatArray,
    rho: float,
    target: float,
    n_iter: int = 48,
    tol: float = 1e-5,
) -> float:
    """Binary search for bandwidth ``sigma`` solving sum(exp(-(d-rho)/sigma)) = target."""
    lo = 1e-6
    hi = 1e6
    mid = 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        val = float(np.sum(np.exp(-np.maximum(distances - rho, 0.0) / mid)))
        if abs(val - target) < tol:
            break
        if val > target:
            hi = mid
        else:
            lo = mid
    return max(mid, 1e-6)


def compute_spectral_embedding(
    connectivities: FloatArray,
    n_components: int = 2,
) -> FloatArray:
    """Compute normalized symmetric Laplacian eigenvectors for manifold initialization."""
    n = connectivities.shape[0]
    w = np.maximum(connectivities, 0.0).copy()
    np.fill_diagonal(w, 0.0)
    deg = np.sum(w, axis=1)
    # Add tiny self-loop to isolated nodes so D^-1/2 is well-defined
    isolated = deg < 1e-12
    if np.any(isolated):
        w[isolated, isolated] = 1.0
        deg = np.sum(w, axis=1)

    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    norm_adj = (d_inv_sqrt[:, None] * w) * d_inv_sqrt[None, :]
    laplacian = np.eye(n, dtype=np.float64) - norm_adj
    laplacian = 0.5 * (laplacian + laplacian.T)

    eigvals, eigvecs = np.linalg.eigh(laplacian)
    order = np.argsort(eigvals)
    # Skip trivial constant eigenvector(s) near eigenvalue 0
    non_trivial = [int(idx) for idx in order if eigvals[idx] > 1e-8]
    if len(non_trivial) < n_components:
        non_trivial = [int(idx) for idx in order[1 : n_components + 1]]

    chosen = non_trivial[:n_components]
    coords = eigvecs[:, chosen].copy()
    for j in range(coords.shape[1]):
        max_idx = int(np.argmax(np.abs(coords[:, j])))
        if coords[max_idx, j] < 0.0:
            coords[:, j] *= -1.0
        std_j = float(np.std(coords[:, j]))
        if std_j > 1e-12:
            coords[:, j] /= std_j
    return coords


def magic_diffusion_impute(
    dataset: TranscriptomeDataset,
    graph: NeighborGraph,
    t_steps: int = 3,
    layers: Tuple[str, ...] = ("spliced", "unspliced"),
) -> dict[str, FloatArray]:
    """Perform Markov transition power diffusion smoothing (MAGIC-style).

    Computes ``P = D^{-1} W`` (with small self-reinforcement) and evaluates
    ``P^t @ M`` for each requested layer, storing denoised layers ``Ms`` and
    ``Mu`` (or ``imputed_<layer>``) inside ``dataset.layers``.
    """
    w = graph.connectivities.copy()
    np.fill_diagonal(w, 1.0)
    row_sums = np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
    p_mat = w / row_sums

    p_power = np.linalg.matrix_power(p_mat, max(1, t_steps))
    results: dict[str, FloatArray] = {}
    for layer_name in layers:
        if layer_name not in dataset.layers:
            raise KeyError(f"Layer '{layer_name}' not found for diffusion imputation")
        smoothed = p_power @ dataset.layers[layer_name]
        smoothed = np.maximum(smoothed, 0.0)
        out_name = (
            "Ms" if layer_name == "spliced" else ("Mu" if layer_name == "unspliced" else f"imputed_{layer_name}")
        )
        dataset.set_layer(out_name, smoothed, allow_negative=False)
        results[out_name] = smoothed
    return results


def compute_umap_layout(
    graph: NeighborGraph,
    n_components: int = 2,
    n_epochs: int = 160,
    learning_rate: float = 0.8,
    min_dist: float = 0.25,
    negative_sample_rate: int = 5,
    seed: int = 42,
) -> FloatArray:
    """Compute a 2D manifold layout using fuzzy cross-entropy SGD optimization.

    Minimizes the fuzzy simplicial cross-entropy between high-dimensional
    edge weights ``w_{ij}`` and low-dimensional Student-t kernel probabilities
    ``q_{ij} = (1 + a * ||y_i - y_j||^{2b})^{-1}`` using spectral initialization.
    """
    n_cells = graph.n_cells
    rng = np.random.default_rng(seed)

    # Fit smooth Student-t curve parameters (a, b) from min_dist
    a_param = 1.58 * (1.0 / max(min_dist, 0.05)) ** 0.22
    b_param = 0.89

    embedding = compute_spectral_embedding(graph.connectivities, n_components=n_components)
    embedding = embedding * 4.0 + rng.normal(0.0, 0.05, size=embedding.shape)

    rows, cols = np.nonzero(np.triu(graph.connectivities, k=1) > 1e-4)
    if len(rows) == 0:
        return embedding

    weights = graph.connectivities[rows, cols]
    # Normalize edge sampling schedule
    max_w = float(np.max(weights))
    norm_w = weights / max(max_w, 1e-12)

    for epoch in range(n_epochs):
        alpha = learning_rate * (1.0 - float(epoch) / float(n_epochs))
        alpha = max(alpha, 0.02)

        # Attractive updates along weighted edges
        active = rng.random(len(rows)) < norm_w
        if np.any(active):
            r_act = rows[active]
            c_act = cols[active]
            diff = embedding[r_act] - embedding[c_act]
            dist_sq = np.sum(diff * diff, axis=1) + 1e-6
            grad_coeff = (-2.0 * a_param * b_param * np.power(dist_sq, b_param - 1.0)) / (
                1.0 + a_param * np.power(dist_sq, b_param)
            )
            step = np.clip(grad_coeff[:, None] * diff, -4.0, 4.0) * alpha
            np.add.at(embedding, r_act, step)
            np.add.at(embedding, c_act, -step)

        # Repulsive negative sampling updates
        n_rep = min(n_cells * negative_sample_rate, 2500)
        src = rng.integers(0, n_cells, size=n_rep)
        dst = rng.integers(0, n_cells, size=n_rep)
        valid_rep = src != dst
        if np.any(valid_rep):
            s_v = src[valid_rep]
            d_v = dst[valid_rep]
            diff_r = embedding[s_v] - embedding[d_v]
            dist_sq_r = np.sum(diff_r * diff_r, axis=1) + 1e-4
            rep_coeff = (2.0 * b_param) / (
                (0.001 + dist_sq_r) * (1.0 + a_param * np.power(dist_sq_r, b_param))
            )
            step_r = np.clip(rep_coeff[:, None] * diff_r, -4.0, 4.0) * (alpha * 0.5)
            np.add.at(embedding, s_v, step_r)

    embedding -= np.mean(embedding, axis=0, keepdims=True)
    return embedding


def compute_force_directed_layout(
    graph: NeighborGraph,
    n_iter: int = 120,
    seed: int = 42,
) -> FloatArray:
    """Compute a 2D Fruchterman-Reingold force-directed graph layout."""
    n_cells = graph.n_cells
    rng = np.random.default_rng(seed)
    pos = compute_spectral_embedding(graph.connectivities, n_components=2)
    pos = pos + rng.normal(0.0, 0.02, size=pos.shape)

    w = graph.connectivities
    k_opt = np.sqrt(1.0 / max(n_cells, 1)) * 2.5

    for it in range(n_iter):
        temp = 0.4 * (1.0 - float(it) / float(n_iter)) + 0.01
        disp = np.zeros_like(pos)

        # Pairwise repulsion (vectorized for modest cell counts)
        diff = pos[:, None, :] - pos[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2) + 1e-6
        dist_for_rep = dist_sq.copy()
        np.fill_diagonal(dist_for_rep, np.inf)
        rep = (k_opt ** 2) / dist_for_rep
        disp += np.sum(diff * rep[:, :, None], axis=1)

        # Graph edge attraction
        att = (w * np.sqrt(dist_sq)) / max(k_opt, 1e-6)
        np.fill_diagonal(att, 0.0)
        disp -= np.sum(diff * att[:, :, None], axis=1)

        disp_norm = np.maximum(np.linalg.norm(disp, axis=1, keepdims=True), 1e-8)
        pos += (disp / disp_norm) * np.minimum(disp_norm, temp)

    pos -= np.mean(pos, axis=0, keepdims=True)
    return pos


def embed_manifold(
    dataset: TranscriptomeDataset,
    n_neighbors: int = 15,
    metric: str = "euclidean",
    layout: str = "umap",
    seed: int = 42,
) -> Tuple[NeighborGraph, FloatArray]:
    """Build kNN graph from ``obsm['X_pca']`` (or ``X``) and store 2D layout."""
    feats = dataset.obsm["X_pca"] if "X_pca" in dataset.obsm else dataset.X
    graph = build_knn_graph(feats, n_neighbors=n_neighbors, metric=metric)
    dataset.set_obsp("connectivities", graph.connectivities)
    dataset.uns["neighbors"] = {
        "n_neighbors": graph.n_neighbors,
        "metric": graph.metric,
    }

    if layout == "umap":
        coords = compute_umap_layout(graph, seed=seed)
        dataset.set_obsm("X_umap", coords)
    elif layout == "force":
        coords = compute_force_directed_layout(graph, seed=seed)
        dataset.set_obsm("X_draw_graph", coords)
    else:
        raise ValueError(f"Unsupported layout '{layout}'; expected 'umap' or 'force'")

    return graph, coords
