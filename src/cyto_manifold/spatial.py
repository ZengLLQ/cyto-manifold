"""Spatial transcriptomics neighborhood graphs, Moran's I, LISA, and L-R signaling.

Implements spatial statistical mechanics and paracrine signaling inference:

1. Spatial graph construction via Delaunay triangulation, Gaussian RBF radius
   cutoff, or Euclidean k-nearest neighbors.
2. Exact global Moran's I and Geary's C spatial autocorrelation with analytical
   randomization expectation ``E[I] = -1 / (N - 1)``, analytical variance
   ``Var(I)``, two-sided normal z-tests, and Benjamini-Hochberg FDR to detect
   Spatially Variable Genes (SVGs).
3. Local Indicators of Spatial Association (LISA / Local Moran's I_i) for
   hotspot/coldspot/boundary zone classification.
4. Spatial niche domain clustering combining cellular expression clusters with
   local spatial neighborhood composition vectors.
5. Curated 24-pair Ligand-Receptor interaction database with spatially gated
   paracrine signaling scoring and empirical cluster-permutation p-values.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.stats import norm

from cyto_manifold.clustering import leiden_cluster
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.types import (
    FloatArray,
    IntArray,
    LigandReceptorPair,
    SignalingInteraction,
    SpatialAnalysisResult,
    SpatialGeneStat,
)

CANONICAL_LIGAND_RECEPTORS: Tuple[LigandReceptorPair, ...] = (
    LigandReceptorPair("WNT3A", "FZD7", "WNT", "Secreted"),
    LigandReceptorPair("WNT5A", "FZD5", "WNT", "Secreted"),
    LigandReceptorPair("RSPO3", "LGR5", "WNT", "Secreted"),
    LigandReceptorPair("DLL1", "NOTCH1", "NOTCH", "Cell-Cell Contact"),
    LigandReceptorPair("JAG1", "NOTCH2", "NOTCH", "Cell-Cell Contact"),
    LigandReceptorPair("TGFB1", "TGFBR1", "TGFb", "Secreted"),
    LigandReceptorPair("BMP4", "BMPR1A", "BMP", "Secreted"),
    LigandReceptorPair("VEGFA", "KDR", "VEGF", "Secreted"),
    LigandReceptorPair("VEGFB", "FLT1", "VEGF", "Secreted"),
    LigandReceptorPair("EGF", "EGFR", "EGF", "Secreted"),
    LigandReceptorPair("AREG", "ERBB2", "EGF", "Secreted"),
    LigandReceptorPair("PDGFA", "PDGFRA", "PDGF", "Secreted"),
    LigandReceptorPair("FGF2", "FGFR1", "FGF", "Secreted"),
    LigandReceptorPair("CXCL12", "CXCR4", "CXCL", "Chemokine"),
    LigandReceptorPair("CCL2", "CCR2", "CCL", "Chemokine"),
    LigandReceptorPair("CCL5", "CCR5", "CCL", "Chemokine"),
    LigandReceptorPair("IL6", "IL6R", "IL6", "Cytokine"),
    LigandReceptorPair("CSF1", "CSF1R", "CSF", "Cytokine"),
    LigandReceptorPair("TNF", "TNFRSF1A", "TNF", "Cytokine"),
    LigandReceptorPair("SHH", "PTCH1", "HH", "Secreted"),
    LigandReceptorPair("COL1A1", "ITGB1", "ECM-Receptor", "ECM"),
    LigandReceptorPair("FN1", "ITGA5", "ECM-Receptor", "ECM"),
    LigandReceptorPair("LGALS9", "HAVCR2", "Checkpoint", "Immune"),
    LigandReceptorPair("CD274", "PDCD1", "Checkpoint", "Immune"),
)


def benjamini_hochberg(p_values: Sequence[float] | FloatArray) -> FloatArray:
    """Apply Benjamini-Hochberg step-up FDR correction."""
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    ranks = np.arange(1, n + 1, dtype=np.float64)
    raw_fdr = sorted_p * (float(n) / ranks)
    # Enforce monotonicity from largest p-value down
    monotonic = np.minimum.accumulate(raw_fdr[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.clip(monotonic, 0.0, 1.0)
    return out


def build_spatial_graph(
    coords: FloatArray,
    mode: str = "delaunay",
    n_neighbors: int = 6,
    radius: Optional[float] = None,
) -> FloatArray:
    """Construct a row-standardized or symmetric spatial weight matrix ``W``.

    Parameters
    ----------
    coords:
        Spatial spot/cell positions of shape ``(n_cells, >=2)``.
    mode:
        ``"delaunay"`` (Delaunay triangulation with Gaussian distance decay),
        ``"knn"`` (k-nearest spatial neighbors), or ``"radius"`` (distance cutoff).
    n_neighbors:
        Neighbor count when ``mode="knn"``.
    radius:
        Distance threshold when ``mode="radius"`` (defaults to 2x median NN distance).
    """
    pts = np.asarray(coords[:, :2], dtype=np.float64)
    n_cells = pts.shape[0]
    if n_cells < 4:
        raise ValueError("Need at least 4 spatial spots to build a spatial graph")

    tree = cKDTree(pts)
    nn_dists, nn_idx = tree.query(pts, k=min(n_neighbors + 1, n_cells))
    median_nn = float(np.median(nn_dists[:, 1]))
    bandwidth = max(median_nn * 1.5, 1e-6)

    w = np.zeros((n_cells, n_cells), dtype=np.float64)

    if mode == "delaunay":
        # Perturb duplicate coordinates by infinitesimal jitter if collinear
        tri = Delaunay(pts + 1e-9 * np.arange(n_cells * 2).reshape(n_cells, 2))
        for simplex in tri.simplices:
            for a, b in ((0, 1), (1, 2), (0, 2)):
                u, v = int(simplex[a]), int(simplex[b])
                dist = float(np.linalg.norm(pts[u] - pts[v]))
                weight = float(np.exp(-(dist ** 2) / (2.0 * bandwidth ** 2)))
                w[u, v] = weight
                w[v, u] = weight
    elif mode == "knn":
        k_eff = min(n_neighbors, n_cells - 1)
        for i in range(n_cells):
            for j_pos in range(1, k_eff + 1):
                j = int(nn_idx[i, j_pos])
                d = float(nn_dists[i, j_pos])
                weight = float(np.exp(-(d ** 2) / (2.0 * bandwidth ** 2)))
                w[i, j] = weight
                w[j, i] = weight
    elif mode == "radius":
        r_cut = float(radius) if radius is not None else max(median_nn * 2.2, 1e-6)
        pairs = tree.query_pairs(r=r_cut)
        for u, v in pairs:
            d = float(np.linalg.norm(pts[u] - pts[v]))
            weight = float(np.exp(-(d ** 2) / (2.0 * bandwidth ** 2)))
            w[u, v] = weight
            w[v, u] = weight
    else:
        raise ValueError(f"Unsupported spatial graph mode '{mode}'")

    np.fill_diagonal(w, 0.0)
    return w


def compute_morans_i(
    values: FloatArray,
    weights: FloatArray,
) -> Tuple[float, float, float, float]:
    """Compute exact global Moran's I, expected E[I], normal z-score, and two-sided p-value.

    Uses the standard randomization variance formula (Cliff & Ord, 1981):
        I = (N / S_0) * (z^T W z) / (z^T z)
    """
    x = np.asarray(values, dtype=np.float64)
    n = float(len(x))
    if n < 4:
        return 0.0, 0.0, 0.0, 1.0

    z = x - np.mean(x)
    m2 = float(np.dot(z, z))
    w = 0.5 * (weights + weights.T)
    s0 = float(np.sum(w))
    if m2 <= 1e-12 or s0 <= 1e-12:
        return 0.0, -1.0 / (n - 1.0), 0.0, 1.0

    moran_i = (n / s0) * float(z @ w @ z) / m2
    e_i = -1.0 / (n - 1.0)

    # Analytical randomization variance
    s1 = 0.5 * float(np.sum((w + w.T) ** 2))
    row_sums = np.sum(w, axis=1)
    s2 = float(np.sum((2.0 * row_sums) ** 2))
    m4 = float(np.sum(z ** 4))
    b2 = (n * m4) / max(m2 ** 2, 1e-18)

    num1 = n * ((n * n - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s0 * s0)
    num2 = b2 * ((n * n - n) * s1 - 2.0 * n * s2 + 6.0 * s0 * s0)
    denom = (n - 1.0) * (n - 2.0) * (n - 3.0) * (s0 * s0)
    var_i = max((num1 - num2) / max(denom, 1e-18) - e_i * e_i, 1e-12)

    z_score = (moran_i - e_i) / float(np.sqrt(var_i))
    p_val = float(2.0 * norm.sf(abs(z_score)))
    return float(moran_i), float(e_i), float(z_score), p_val


def compute_gearys_c(
    values: FloatArray,
    weights: FloatArray,
) -> Tuple[float, float]:
    """Compute Geary's C local spatial autocorrelation (C < 1 indicates positive clustering)."""
    x = np.asarray(values, dtype=np.float64)
    n = float(len(x))
    z = x - np.mean(x)
    m2 = float(np.dot(z, z))
    w = 0.5 * (weights + weights.T)
    s0 = float(np.sum(w))
    if n < 4 or m2 <= 1e-12 or s0 <= 1e-12:
        return 1.0, 0.0

    diff_sq = (x[:, None] - x[None, :]) ** 2
    num = (n - 1.0) * float(np.sum(w * diff_sq))
    denom = 2.0 * s0 * m2
    c_val = num / max(denom, 1e-12)

    # Approximate z-score under normality assumption
    s1 = 0.5 * float(np.sum((w + w.T) ** 2))
    row_sums = np.sum(w, axis=1)
    s2 = float(np.sum((2.0 * row_sums) ** 2))
    var_c = ((2.0 * s1 + s2) * (n - 1.0) - 4.0 * s0 * s0) / max(2.0 * (n + 1.0) * s0 * s0, 1e-12)
    z_c = (1.0 - c_val) / float(np.sqrt(max(var_c, 1e-12)))
    return float(c_val), float(z_c)


def compute_local_moran(
    values: FloatArray,
    weights: FloatArray,
) -> FloatArray:
    """Compute Local Indicators of Spatial Association (LISA / Local Moran's I_i)."""
    x = np.asarray(values, dtype=np.float64)
    z = x - np.mean(x)
    m2 = float(np.mean(z * z))
    if m2 <= 1e-12:
        return np.zeros_like(x)
    row_sums = np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
    w_row = weights / row_sums
    lag_z = w_row @ z
    return (z / m2) * lag_z


def detect_spatially_variable_genes(
    dataset: TranscriptomeDataset,
    weights: FloatArray,
    fdr_threshold: float = 0.05,
) -> List[SpatialGeneStat]:
    """Score every gene in ``dataset`` with Moran's I, Geary's C, and BH-FDR."""
    expr = dataset.X
    stats: List[Tuple[str, int, float, float, float, float, float, float]] = []
    p_vals: List[float] = []

    for j, g_name in enumerate(dataset.gene_names):
        col = expr[:, j]
        mi, ei, zi, pi = compute_morans_i(col, weights)
        gc, zc = compute_gearys_c(col, weights)
        stats.append((g_name, j, mi, ei, zi, pi, gc, zc))
        p_vals.append(pi)

    fdrs = benjamini_hochberg(p_vals)
    results: List[SpatialGeneStat] = []
    for (g_name, j, mi, ei, zi, pi, gc, zc), fdr in zip(stats, fdrs):
        is_svg = (fdr <= fdr_threshold) and (mi > ei)
        results.append(
            SpatialGeneStat(
                gene_name=g_name,
                gene_index=j,
                morans_i=mi,
                expected_i=ei,
                z_score_i=zi,
                p_value_i=pi,
                fdr_i=float(fdr),
                gearys_c=gc,
                z_score_c=zc,
                is_spatially_variable=is_svg,
            )
        )

    results.sort(key=lambda r: (-r.z_score_i, r.fdr_i))
    return results


def identify_spatial_niches(
    dataset: TranscriptomeDataset,
    weights: FloatArray,
    labels: Optional[IntArray] = None,
    resolution: float = 0.8,
    seed: int = 42,
) -> IntArray:
    """Identify spatial tissue niches via expression-spatial neighborhood coupling."""
    w_norm = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
    if labels is None:
        labels = (
            np.asarray(dataset.obs["cluster"], dtype=np.int64)
            if "cluster" in dataset.obs
            else np.zeros(dataset.n_cells, dtype=np.int64)
        )
    # Build cellular composition vector in each cell's spatial neighborhood
    unique_c = np.unique(labels)
    c_map = {int(c): i for i, c in enumerate(unique_c)}
    one_hot = np.zeros((dataset.n_cells, len(unique_c)), dtype=np.float64)
    for idx, c in enumerate(labels):
        one_hot[idx, c_map[int(c)]] = 1.0
    niche_comp = 0.5 * one_hot + 0.5 * (w_norm @ one_hot)
    sim = niche_comp @ niche_comp.T
    coupled_adj = weights * np.maximum(sim, 0.05)
    np.fill_diagonal(coupled_adj, 0.0)
    niche_labels = leiden_cluster(coupled_adj, resolution=resolution, seed=seed)
    dataset.obs["spatial_niche"] = niche_labels
    return niche_labels


def score_ligand_receptor_signaling(
    dataset: TranscriptomeDataset,
    weights: FloatArray,
    labels: IntArray,
    pairs: Sequence[LigandReceptorPair] = CANONICAL_LIGAND_RECEPTORS,
    n_permutations: int = 500,
    fdr_threshold: float = 0.05,
    min_spatial_edges: int = 2,
    seed: int = 42,
) -> List[SignalingInteraction]:
    """Compute spatially gated ligand-receptor cell-cell communication with permutations.

    For each cluster pair ``(A, B)`` linked by spatial edges ``W_{ij} > 0``:
        ``S(A -> B) = sum_{i in A, j in B} W_{ij} * L_i * R_j / sum_{i in A, j in B} 1_{W_{ij}>0}``
    Uses vectorized indicator matrix contractions ``M^T @ P @ M`` to evaluate
    hundreds of cluster-label permutations efficiently, followed by BH-FDR.
    """
    gene_map = {g: i for i, g in enumerate(dataset.gene_names)}
    present_pairs = [p for p in pairs if p.ligand in gene_map and p.receptor in gene_map]
    if not present_pairs:
        return []

    expr = dataset.X
    w_bin = (weights > 1e-6).astype(np.float64)
    unique_clusters = sorted(int(c) for c in np.unique(labels))
    k_clust = len(unique_clusters)
    c_to_col = {cid: idx for idx, cid in enumerate(unique_clusters)}
    m_obs = np.zeros((dataset.n_cells, k_clust), dtype=np.float64)
    for cell_idx, cid in enumerate(labels):
        m_obs[cell_idx, c_to_col[int(cid)]] = 1.0

    obs_edges_mat = m_obs.T @ w_bin @ m_obs
    n_perms = max(20, n_permutations)
    rng = np.random.default_rng(seed)
    perm_mats = [m_obs[rng.permutation(dataset.n_cells)] for _ in range(n_perms)]
    perm_edges_mats = [pm.T @ w_bin @ pm for pm in perm_mats]

    raw_results: List[Tuple[int, int, LigandReceptorPair, float, float, int]] = []
    p_vals: List[float] = []

    for pair in present_pairs:
        lig = expr[:, gene_map[pair.ligand]]
        rec = expr[:, gene_map[pair.receptor]]
        if float(np.max(lig)) <= 1e-8 or float(np.max(rec)) <= 1e-8:
            continue

        pair_mat = (weights * lig[:, None]) * rec[None, :]
        obs_sum = m_obs.T @ pair_mat @ m_obs
        obs_scores = np.divide(
            obs_sum,
            np.maximum(obs_edges_mat, 1.0),
            out=np.zeros_like(obs_sum),
            where=obs_edges_mat > 0.0,
        )

        # Evaluate permutations across all (sender, receiver) pairs simultaneously
        exceed_counts = np.zeros((k_clust, k_clust), dtype=np.int64)
        valid_counts = np.zeros((k_clust, k_clust), dtype=np.int64)
        for pm, pe in zip(perm_mats, perm_edges_mats):
            p_sum = pm.T @ pair_mat @ pm
            p_scores = np.divide(
                p_sum,
                np.maximum(pe, 1.0),
                out=np.zeros_like(p_sum),
                where=pe > 0.0,
            )
            has_e = pe > 0.0
            valid_counts += has_e.astype(np.int64)
            exceed_counts += (has_e & (p_scores >= obs_scores - 1e-12)).astype(np.int64)

        for i_src, c_src in enumerate(unique_clusters):
            for j_dst, c_dst in enumerate(unique_clusters):
                n_edges = int(obs_edges_mat[i_src, j_dst])
                if n_edges < min_spatial_edges:
                    continue
                score = float(obs_scores[i_src, j_dst])
                if score <= 1e-8:
                    continue
                v_cnt = max(int(valid_counts[i_src, j_dst]), 1)
                ex_cnt = int(exceed_counts[i_src, j_dst])
                p_val = float(ex_cnt + 1) / float(v_cnt + 1)
                raw_results.append((c_src, c_dst, pair, score, p_val, n_edges))
                p_vals.append(p_val)

    fdrs = benjamini_hochberg(p_vals)
    interactions: List[SignalingInteraction] = []
    for (c_src, c_dst, pair, score, p_val, n_edges), fdr in zip(raw_results, fdrs):
        interactions.append(
            SignalingInteraction(
                sender_cluster=c_src,
                receiver_cluster=c_dst,
                ligand=pair.ligand,
                receptor=pair.receptor,
                pathway=pair.pathway,
                score=score,
                p_value=p_val,
                fdr=float(fdr),
                n_spatial_edges=n_edges,
                significant=bool(fdr <= fdr_threshold),
            )
        )

    interactions.sort(key=lambda s: (-int(s.significant), -s.score, s.fdr))
    return interactions


def analyze_spatial_tissue(
    dataset: TranscriptomeDataset,
    mode: str = "delaunay",
    n_neighbors: int = 6,
    n_permutations: int = 500,
    fdr_threshold: float = 0.05,
    seed: int = 42,
) -> SpatialAnalysisResult:
    """Run complete spatial graph, SVG, LISA hotspot, niche, and L-R signaling analysis."""
    if "spatial" not in dataset.obsm:
        raise ValueError("dataset.obsm['spatial'] coordinates are required for spatial analysis")

    weights = build_spatial_graph(
        dataset.obsm["spatial"], mode=mode, n_neighbors=n_neighbors
    )
    dataset.set_obsp("spatial_weights", weights)

    gene_stats = detect_spatially_variable_genes(
        dataset, weights, fdr_threshold=fdr_threshold
    )
    n_svg = sum(1 for g in gene_stats if g.is_spatially_variable)

    # Compute LISA for the top spatially variable gene
    top_idx = gene_stats[0].gene_index if gene_stats else 0
    lisa = compute_local_moran(dataset.X[:, top_idx], weights)
    dataset.obs["top_svg_lisa"] = lisa

    labels = (
        np.asarray(dataset.obs["cluster"], dtype=np.int64)
        if "cluster" in dataset.obs
        else np.zeros(dataset.n_cells, dtype=np.int64)
    )
    niches = identify_spatial_niches(dataset, weights, labels=labels, seed=seed)
    signaling = score_ligand_receptor_signaling(
        dataset,
        weights,
        labels=labels,
        n_permutations=n_permutations,
        fdr_threshold=fdr_threshold,
        seed=seed,
    )

    return SpatialAnalysisResult(
        spatial_weights=weights,
        gene_stats=gene_stats,
        n_spatially_variable=n_svg,
        local_moran=lisa,
        niche_labels=niches,
        signaling_interactions=signaling,
        graph_mode=mode,
    )
