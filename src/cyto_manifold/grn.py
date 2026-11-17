"""Gene Regulatory Network (GRN) inference, CLR normalization, and AUCell scoring.

Combines information-theoretic co-expression with transcriptional splicing
causality and single-cell regulon activity scoring:

1. Quantile-discretized Mutual Information (MI) matrix ``I(TF; Target)``
   and Context Likelihood of Relatedness (CLR) background z-score
   normalization (Faith et al., 2007).
2. Ridge-regularized TF-to-target linear regression weights paired with
   splicing-lagged causal velocity correlation ``corr(u_target, s_TF)`` to
   orient activator/repressor directionality.
3. AUCell-style rank-recovery Area Under the Curve (AUC) regulon activity
   scoring per cell (Aibar et al., 2017).
4. Regulatory network topology metrics: out-degree hub centrality and
   feedforward loop (FFL) triad counting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.types import FloatArray, GRNResult, RegulonEdge


def _quantile_discretize(vec: FloatArray, n_bins: int = 8) -> np.ndarray:
    """Discretize a continuous vector into quantile bins handling ties."""
    ranks = np.argsort(np.argsort(vec, kind="mergesort"))
    bins = (ranks * n_bins) // max(len(vec), 1)
    return np.clip(bins, 0, n_bins - 1).astype(np.int64)


def compute_mutual_information(x: FloatArray, y: FloatArray, n_bins: int = 8) -> float:
    """Compute discrete Shannon Mutual Information ``I(X; Y)`` in nats."""
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    bx = _quantile_discretize(x, n_bins=n_bins)
    by = _quantile_discretize(y, n_bins=n_bins)
    joint = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(joint, (bx, by), 1.0)
    p_xy = joint / max(float(len(x)), 1.0)
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    nz = p_xy > 0.0
    denom = (p_x @ p_y)[nz]
    mi = float(np.sum(p_xy[nz] * np.log(p_xy[nz] / np.maximum(denom, 1e-12))))
    return max(mi, 0.0)


def compute_clr_matrix(mi_matrix: FloatArray) -> FloatArray:
    """Apply Context Likelihood of Relatedness (CLR) background z-score."""
    n_tf, n_genes = mi_matrix.shape
    row_means = np.mean(mi_matrix, axis=1, keepdims=True)
    row_stds = np.maximum(np.std(mi_matrix, axis=1, ddof=1, keepdims=True), 1e-8)
    col_means = np.mean(mi_matrix, axis=0, keepdims=True)
    col_stds = np.maximum(np.std(mi_matrix, axis=0, ddof=1, keepdims=True), 1e-8)

    z_row = np.maximum((mi_matrix - row_means) / row_stds, 0.0)
    z_col = np.maximum((mi_matrix - col_means) / col_stds, 0.0)
    clr = np.sqrt(0.5 * (z_row ** 2 + z_col ** 2))
    return np.asarray(clr, dtype=np.float64)


def compute_aucell_scores(
    expr: FloatArray,
    gene_names: Sequence[str],
    regulons: Dict[str, List[str]],
    auc_top_fraction: float = 0.25,
) -> Tuple[FloatArray, List[str]]:
    """Score per-cell regulon activity via rank-recovery AUC (AUCell)."""
    n_cells, n_genes = expr.shape
    tf_names = sorted(regulons.keys())
    if not tf_names:
        return np.zeros((n_cells, 0), dtype=np.float64), []

    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_top = max(3, min(n_genes, int(round(n_genes * auc_top_fraction))))

    # Rank genes descending per cell (0 = highest expressed)
    order_desc = np.argsort(-expr, axis=1, kind="mergesort")
    ranks = np.empty_like(order_desc)
    row_grid = np.arange(n_cells)[:, None]
    ranks[row_grid, order_desc] = np.arange(n_genes)[None, :]

    auc_mat = np.zeros((n_cells, len(tf_names)), dtype=np.float64)
    for col_idx, tf in enumerate(tf_names):
        targets = [gene_to_idx[g] for g in regulons[tf] if g in gene_to_idx]
        if not targets:
            continue
        t_ranks = ranks[:, targets]  # (n_cells, n_targets)
        # Exact discrete recovery curve area over [0, n_top - 1]
        clipped = np.minimum(t_ranks, n_top)
        auc_raw = np.sum(n_top - clipped, axis=1).astype(np.float64)
        max_area = float(len(targets) * n_top)
        auc_mat[:, col_idx] = auc_raw / max(max_area, 1.0)

    return auc_mat, tf_names


def _count_feedforward_triads(edges: Sequence[RegulonEdge]) -> int:
    """Count directed 3-node feedforward loops (A->B, B->C, A->C)."""
    adj: Dict[str, set[str]] = {}
    for e in edges:
        adj.setdefault(e.tf_name, set()).add(e.target_name)
    triads = 0
    for a, a_targets in adj.items():
        for b in a_targets:
            if b not in adj or b == a:
                continue
            b_targets = adj[b]
            for c in b_targets:
                if c != a and c != b and c in a_targets:
                    triads += 1
    return triads


def infer_gene_regulatory_network(
    dataset: TranscriptomeDataset,
    tf_names: Optional[Sequence[str]] = None,
    ridge_lambda: float = 2.0,
    clr_threshold: float = 1.2,
    n_bins: int = 8,
    max_edges: int = 120,
) -> GRNResult:
    """Infer directed TF-target GRN edges, hub centralities, and AUCell scores.

    Parameters
    ----------
    dataset:
        Normalized ``TranscriptomeDataset`` (uses ``unspliced`` if available
        for splicing-lagged causal correlation).
    tf_names:
        Candidate transcription factors. Defaults to top high-variance genes
        or genes prefixed with ``TF_`` / known canonical TFs.
    ridge_lambda:
        L2 Ridge penalty for multivariate TF -> target regression.
    clr_threshold:
        Minimum CLR z-score required to retain a regulatory edge.
    n_bins:
        Quantile bin count for Shannon Mutual Information estimation.
    max_edges:
        Maximum directed edges retained in the returned network.
    """
    expr = dataset.X
    n_cells, n_genes = expr.shape
    gene_names = dataset.gene_names
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    if tf_names is not None:
        tfs = [g for g in tf_names if g in gene_to_idx]
    else:
        canonical_tfs = {
            "SOX2", "SOX9", "NANOG", "OCT4", "MYC", "TP53", "FOXO3", "SNAI1",
            "ZEB1", "TWIST1", "GATA3", "RUNX1", "STAT3", "HIF1A", "NFKB1",
            "E2F1", "HES1", "LEF1", "TCF7", "SMAD3", "ETS1", "IRF8", "SPI1",
        }
        tfs = [g for g in gene_names if g in canonical_tfs or g.startswith("TF_")]
        if len(tfs) < 4:
            # Select top-variance genes as putative regulators
            variances = np.var(expr, axis=0)
            top_k = min(10, max(3, n_genes // 5))
            top_idx = np.argsort(-variances, kind="mergesort")[:top_k]
            tfs = [gene_names[int(i)] for i in top_idx]

    tf_indices = [gene_to_idx[t] for t in tfs]
    n_tf = len(tf_indices)

    # 1. Compute pairwise MI(TF, Target) and CLR background normalization
    mi_mat = np.zeros((n_tf, n_genes), dtype=np.float64)
    for ti, g_tf in enumerate(tf_indices):
        x_tf = expr[:, g_tf]
        for gj in range(n_genes):
            if g_tf == gj:
                continue
            mi_mat[ti, gj] = compute_mutual_information(x_tf, expr[:, gj], n_bins=n_bins)
    clr_mat = compute_clr_matrix(mi_mat)

    # 2. Multivariate Ridge regression from standardized TFs -> standardized targets
    x_tf_mat = expr[:, tf_indices]
    x_mean = np.mean(x_tf_mat, axis=0, keepdims=True)
    x_std = np.maximum(np.std(x_tf_mat, axis=0, ddof=1, keepdims=True), 1e-8)
    x_z = (x_tf_mat - x_mean) / x_std

    y_mean = np.mean(expr, axis=0, keepdims=True)
    y_std = np.maximum(np.std(expr, axis=0, ddof=1, keepdims=True), 1e-8)
    y_z = (expr - y_mean) / y_std

    gram = x_z.T @ x_z + ridge_lambda * np.eye(n_tf, dtype=np.float64)
    reg_weights = np.linalg.solve(gram, x_z.T @ y_z)  # (n_tf, n_genes)

    # 3. Velocity/unspliced causal lag correlation corr(u_target, s_TF)
    s_mat = dataset.layers.get("Ms", dataset.layers["spliced"])
    u_mat = dataset.layers.get("Mu", dataset.layers["unspliced"])

    edges: List[RegulonEdge] = []
    for ti, g_tf in enumerate(tf_indices):
        s_tf = s_mat[:, g_tf]
        s_tf_c = s_tf - np.mean(s_tf)
        s_tf_n = float(np.linalg.norm(s_tf_c))
        for gj in range(n_genes):
            if g_tf == gj:
                continue
            clr_val = float(clr_mat[ti, gj])
            mi_val = float(mi_mat[ti, gj])
            w_val = float(reg_weights[ti, gj])
            if clr_val < clr_threshold or abs(w_val) < 0.03:
                continue

            u_tgt = u_mat[:, gj]
            u_c = u_tgt - np.mean(u_tgt)
            u_n = float(np.linalg.norm(u_c))
            if s_tf_n > 1e-12 and u_n > 1e-12:
                lag_corr = float(np.dot(s_tf_c, u_c) / (s_tf_n * u_n))
            else:
                lag_corr = 0.0

            edges.append(
                RegulonEdge(
                    tf_name=tfs[ti],
                    target_name=gene_names[gj],
                    clr_score=clr_val,
                    mutual_information=mi_val,
                    regression_weight=w_val,
                    velocity_lag_corr=lag_corr,
                    direction_sign=1 if w_val >= 0.0 else -1,
                )
            )

    edges.sort(key=lambda e: (-e.clr_score, -abs(e.regression_weight)))
    edges = edges[:max_edges]

    # Build regulon target lists for AUCell
    regulons: Dict[str, List[str]] = {tf: [] for tf in tfs}
    hub_scores: Dict[str, float] = {tf: 0.0 for tf in tfs}
    for e in edges:
        if e.direction_sign > 0:
            regulons[e.tf_name].append(e.target_name)
        hub_scores[e.tf_name] += e.clr_score * abs(e.regression_weight)

    aucell_mat, auc_tfs = compute_aucell_scores(expr, gene_names, regulons)
    dataset.set_obsm("aucell_regulons", aucell_mat)
    triads = _count_feedforward_triads(edges)

    return GRNResult(
        edges=edges,
        aucell_scores=aucell_mat,
        tf_names=auc_tfs,
        hub_centrality=hub_scores,
        n_feedforward_triads=triads,
    )
