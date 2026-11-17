"""Quality control, normalization, HVG selection, PCA, and Harmony batch integration.

Implements standard single-cell transcriptomics preprocessing pipelines in
self-contained NumPy and SciPy:

1. Per-cell and per-gene quality control (total UMI counts, detected genes,
   mitochondrial fraction, unspliced ratio, and Scrublet-style synthetic
   doublet nearest-neighbor scoring).
2. Library-size CP10K normalization and log1p transformation.
3. Analytic Pearson residuals for negative-binomial overdispersion stabilization
   (Lause, Berens, & Kobak, 2021).
4. Variance-stabilizing transformation (VST / Seurat v3 style) HVG selection.
5. Deterministic truncated SVD principal component analysis with variance
   explanation and sign-flipping reproducibility.
6. Harmony-style soft k-means mixture clustering and Ridge centroid alignment
   for multi-batch technical effect removal in PCA space.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.types import (
    FloatArray,
    HVGResult,
    PCAResult,
    QCMetrics,
)


def compute_qc_metrics(
    dataset: TranscriptomeDataset,
    min_genes_per_cell: int = 10,
    min_cells_per_gene: int = 3,
    max_mito_fraction: float = 0.25,
    max_doublet_score: float = 0.85,
    mito_prefixes: Tuple[str, ...] = ("MT-", "mt-"),
    seed: int = 42,
) -> QCMetrics:
    """Compute QC statistics, synthetic doublet scores, and retention masks.

    Parameters
    ----------
    dataset:
        Input ``TranscriptomeDataset`` with raw count matrix ``X``.
    min_genes_per_cell:
        Minimum number of expressed genes (count > 0) required per cell.
    min_cells_per_gene:
        Minimum number of cells expressing a gene required to retain it.
    max_mito_fraction:
        Upper threshold for mitochondrial UMI fraction per cell.
    max_doublet_score:
        Upper threshold on estimated synthetic doublet probability.
    mito_prefixes:
        Gene symbol prefixes identifying mitochondrial transcripts.
    seed:
        Random seed for synthetic doublet simulation.
    """
    counts = dataset.layers["raw_counts"]
    spliced = dataset.layers["spliced"]
    unspliced = dataset.layers["unspliced"]

    total_counts = np.sum(counts, axis=1)
    n_genes_detected = np.sum(counts > 0.0, axis=1).astype(np.int64)
    n_cells_per_gene = np.sum(counts > 0.0, axis=0).astype(np.int64)

    total_sp = np.sum(spliced, axis=1) + np.sum(unspliced, axis=1)
    unspliced_fraction = np.divide(
        np.sum(unspliced, axis=1),
        np.maximum(total_sp, 1e-12),
    )

    mito_mask = np.array(
        [any(g.startswith(pfx) for pfx in mito_prefixes) for g in dataset.gene_names],
        dtype=bool,
    )
    if np.any(mito_mask):
        mito_counts = np.sum(counts[:, mito_mask], axis=1)
        mito_fraction = np.divide(mito_counts, np.maximum(total_counts, 1e-12))
    else:
        mito_fraction = np.zeros(dataset.n_cells, dtype=np.float64)

    doublet_scores = estimate_doublet_scores(counts, seed=seed)

    cell_pass = (
        (n_genes_detected >= min_genes_per_cell)
        & (mito_fraction <= max_mito_fraction)
        & (doublet_scores <= max_doublet_score)
        & (total_counts > 0.0)
    )
    gene_pass = n_cells_per_gene >= min_cells_per_gene

    # Guard against filtering everything on very small synthetic inputs
    if not np.any(cell_pass):
        cell_pass = total_counts > 0.0
    if not np.any(gene_pass):
        gene_pass = np.sum(counts, axis=0) > 0.0

    return QCMetrics(
        total_counts=total_counts,
        n_genes_detected=n_genes_detected,
        unspliced_fraction=unspliced_fraction,
        mito_fraction=mito_fraction,
        doublet_scores=doublet_scores,
        cell_pass_mask=cell_pass,
        gene_pass_mask=gene_pass,
        n_cells_retained=int(np.sum(cell_pass)),
        n_genes_retained=int(np.sum(gene_pass)),
    )


def estimate_doublet_scores(
    counts: FloatArray,
    sim_ratio: float = 0.5,
    n_neighbors: int = 12,
    n_pcs: int = 15,
    seed: int = 42,
) -> FloatArray:
    """Scrublet-style synthetic doublet detection via k-nearest neighbors.

    Simulated doublets are formed by summing random pairs of observed count
    vectors. Both observed and simulated profiles are library-normalized,
    projected via truncated SVD, and scored by local doublet neighbor density
    under Bayes' rule.
    """
    n_cells, n_genes = counts.shape
    if n_cells < 8 or n_genes < 4:
        return np.zeros(n_cells, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n_sim = max(8, int(round(n_cells * sim_ratio)))
    idx_a = rng.integers(0, n_cells, size=n_sim)
    idx_b = rng.integers(0, n_cells, size=n_sim)
    sim_counts = counts[idx_a] + counts[idx_b]

    combined = np.vstack([counts, sim_counts])
    row_sums = np.maximum(np.sum(combined, axis=1, keepdims=True), 1.0)
    norm_log = np.log1p(1e4 * combined / row_sums)
    centered = norm_log - np.mean(norm_log, axis=0, keepdims=True)

    k_pcs = max(2, min(n_pcs, n_cells - 1, n_genes - 1))
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:k_pcs].T

    k_eff = max(3, min(n_neighbors, combined.shape[0] - 1))
    tree = cKDTree(coords)
    _, nn_idx = tree.query(coords[:n_cells], k=k_eff + 1)
    nn_idx = nn_idx[:, 1:]  # drop self

    is_sim = nn_idx >= n_cells
    sim_frac = np.mean(is_sim, axis=1)
    prior = 0.08
    denom = sim_frac + (1.0 - sim_frac) * (sim_ratio / max(prior, 1e-6)) * (1.0 - prior)
    scores = np.divide(sim_frac, np.maximum(denom, 1e-12))
    return np.asarray(np.clip(scores, 0.0, 1.0), dtype=np.float64)


def filter_and_normalize(
    dataset: TranscriptomeDataset,
    qc: Optional[QCMetrics] = None,
    target_sum: float = 1e4,
    inplace: bool = False,
) -> TranscriptomeDataset:
    """Filter cells/genes by QC masks and apply CP10K + log1p normalization.

    Returns a dataset whose ``X`` and ``layers['normalized']`` contain
    log-normalized expression values while preserving ``raw_counts``,
    ``spliced``, and ``unspliced``.
    """
    if qc is None:
        qc = compute_qc_metrics(dataset)

    filtered = dataset.subset(cell_mask=qc.cell_pass_mask, gene_mask=qc.gene_pass_mask)
    raw = filtered.layers["raw_counts"]
    lib_sizes = np.maximum(np.sum(raw, axis=1, keepdims=True), 1.0)
    norm = np.log1p(raw * (target_sum / lib_sizes))

    if inplace:
        dataset.X = norm
        dataset.layers["normalized"] = norm.copy()
        return dataset

    filtered.X = norm
    filtered.layers["normalized"] = norm.copy()
    filtered.obs["total_counts"] = np.sum(raw, axis=1)
    filtered.obs["n_genes_detected"] = np.sum(raw > 0.0, axis=1).astype(np.int64)
    return filtered


def compute_pearson_residuals(
    counts: FloatArray,
    theta: float = 100.0,
    clip_sqrt_n: bool = True,
) -> FloatArray:
    """Compute analytic Negative Binomial Pearson residuals.

    Implements Lause, Berens, & Kobak (2021):
    ``mu_ij = (n_i * g_j) / N``, ``var_ij = mu_ij + mu_ij^2 / theta``,
    and ``z_ij = (X_ij - mu_ij) / sqrt(var_ij)`` clipped to ``[-sqrt(N_c), +sqrt(N_c)]``.
    """
    if theta <= 0.0:
        raise ValueError("Overdispersion theta must be positive")
    n_cells = counts.shape[0]
    row_sums = np.sum(counts, axis=1, keepdims=True)
    col_sums = np.sum(counts, axis=0, keepdims=True)
    total = max(float(np.sum(counts)), 1e-12)
    mu = np.maximum((row_sums @ col_sums) / total, 1e-12)
    variance = mu + (mu ** 2) / theta
    residuals = (counts - mu) / np.sqrt(variance)
    if clip_sqrt_n:
        bound = float(np.sqrt(max(n_cells, 1)))
        residuals = np.clip(residuals, -bound, bound)
    return np.asarray(residuals, dtype=np.float64)


def select_highly_variable_genes(
    dataset: TranscriptomeDataset,
    n_top_genes: int = 40,
    theta: float = 100.0,
    n_bins: int = 20,
) -> HVGResult:
    """Select Highly Variable Genes using VST variance standardization + Pearson residuals.

    Combines log-variance vs. log-mean local trend fitting (Seurat v3 VST)
    with analytic Pearson residual variance to rank informative features
    robustly across dynamic ranges.
    """
    counts = dataset.layers["raw_counts"]
    n_cells, n_genes = counts.shape
    means = np.mean(counts, axis=0)
    variances = np.var(counts, axis=0, ddof=1) if n_cells > 1 else np.zeros(n_genes)

    eps = 1e-8
    log_means = np.log10(means + eps)
    log_vars = np.log10(variances + eps)

    # Local polynomial / binned loess approximation for mean-variance trend
    valid = means > 0.0
    expected_log_vars = log_vars.copy()
    if np.sum(valid) >= 5:
        x_v = log_means[valid]
        y_v = log_vars[valid]
        deg = min(2, max(1, int(np.sum(valid)) - 1))
        coeffs = np.polyfit(x_v, y_v, deg=deg)
        expected_log_vars = np.polyval(coeffs, log_means)
        # Refine with binned local residuals if enough genes exist
        if n_genes >= n_bins:
            edges = np.linspace(float(np.min(x_v)), float(np.max(x_v)) + 1e-9, n_bins + 1)
            for b in range(n_bins):
                in_bin = valid & (log_means >= edges[b]) & (log_means < edges[b + 1])
                if np.sum(in_bin) >= 3:
                    expected_log_vars[in_bin] = float(np.median(log_vars[in_bin]))

    expected_vars = np.maximum((10.0 ** expected_log_vars) - eps, 1e-6)
    std_expected = np.sqrt(expected_vars)
    clip_val = float(np.sqrt(max(n_cells, 1)))
    standardized = (counts - means[None, :]) / std_expected[None, :]
    standardized = np.clip(standardized, -clip_val, clip_val)
    norm_vars = np.var(standardized, axis=0, ddof=1) if n_cells > 1 else np.zeros(n_genes)

    pearson_res = compute_pearson_residuals(counts, theta=theta, clip_sqrt_n=True)
    pearson_vars = np.var(pearson_res, axis=0, ddof=1) if n_cells > 1 else np.zeros(n_genes)

    # Combined rank score
    rank_score = 0.5 * norm_vars + 0.5 * pearson_vars
    k = max(1, min(n_top_genes, n_genes))
    top_indices = np.argsort(-rank_score, kind="mergesort")[:k].astype(np.int64)
    hvg_mask = np.zeros(n_genes, dtype=bool)
    hvg_mask[top_indices] = True

    dataset.var["highly_variable"] = hvg_mask
    dataset.var["vst_variance"] = norm_vars
    dataset.var["pearson_residual_var"] = pearson_vars

    return HVGResult(
        gene_names=list(dataset.gene_names),
        means=means,
        variances=variances,
        expected_variances=expected_vars,
        normalized_variances=norm_vars,
        pearson_residual_vars=pearson_vars,
        highly_variable_mask=hvg_mask,
        top_indices=top_indices,
    )


def compute_pca(
    dataset: TranscriptomeDataset,
    n_components: int = 20,
    use_hvg: bool = True,
    scale_features: bool = True,
    max_abs_val: float = 10.0,
) -> PCAResult:
    """Run deterministic SVD PCA with unit-variance scaling and sign stabilization."""
    mat = dataset.X
    if use_hvg and "highly_variable" in dataset.var:
        hvg_mask = np.asarray(dataset.var["highly_variable"], dtype=bool)
        if np.any(hvg_mask):
            mat = mat[:, hvg_mask]

    n_cells, n_feat = mat.shape
    mean_vec = np.mean(mat, axis=0)
    centered = mat - mean_vec[None, :]

    if scale_features:
        stds = np.std(centered, axis=0, ddof=1) if n_cells > 1 else np.ones(n_feat)
        stds = np.where(stds < 1e-8, 1.0, stds)
        centered = np.clip(centered / stds[None, :], -max_abs_val, max_abs_val)

    k = max(2, min(n_components, n_cells - 1, n_feat))
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    u_k = u[:, :k]
    s_k = s[:k]
    vt_k = vt[:k, :]

    # Deterministic sign convention: largest absolute loading entry per PC is positive
    for j in range(k):
        max_idx = int(np.argmax(np.abs(vt_k[j])))
        if vt_k[j, max_idx] < 0.0:
            vt_k[j] *= -1.0
            u_k[:, j] *= -1.0

    coords = u_k * s_k[None, :]
    denom = max(n_cells - 1, 1)
    explained_var = (s_k ** 2) / denom
    total_var = float(np.sum((s ** 2) / denom))
    explained_ratio = (
        explained_var / max(total_var, 1e-12)
        if total_var > 0.0
        else np.zeros(k, dtype=np.float64)
    )

    dataset.set_obsm("X_pca", coords)
    return PCAResult(
        coordinates=coords,
        loadings=vt_k.T,
        singular_values=s_k,
        explained_variance=explained_var,
        explained_variance_ratio=explained_ratio,
        mean_vector=mean_vec,
        batch_corrected=False,
    )


def integrate_batches_harmony(
    pca: PCAResult,
    batch_labels: Sequence[str],
    n_clusters: Optional[int] = None,
    theta: float = 2.0,
    sigma: float = 0.1,
    ridge_lambda: float = 1.0,
    max_iter: int = 12,
    tol: float = 1e-4,
    seed: int = 42,
) -> PCAResult:
    """Harmony-inspired soft k-means diversity-penalized batch correction.

    Alternates between:
      1. Soft maximum-entropy spherical clustering with a batch-diversity penalty
         (Korsunsky et al., 2019).
      2. Mixture-of-experts Ridge linear batch effect removal across clusters.
    """
    z_orig = pca.coordinates.copy()
    n_cells, n_dims = z_orig.shape
    if len(batch_labels) != n_cells:
        raise ValueError("Length of batch_labels must equal n_cells")

    unique_batches = sorted(set(str(b) for b in batch_labels))
    n_batches = len(unique_batches)
    if n_batches <= 1:
        return pca

    batch_to_idx: Dict[str, int] = {b: i for i, b in enumerate(unique_batches)}
    b_indices = np.array([batch_to_idx[str(b)] for b in batch_labels], dtype=np.int64)
    phi = np.zeros((n_cells, n_batches), dtype=np.float64)
    phi[np.arange(n_cells), b_indices] = 1.0
    batch_priors = np.mean(phi, axis=0)

    k_clust = n_clusters if n_clusters is not None else max(3, min(12, n_cells // 20))
    k_clust = max(2, min(k_clust, n_cells - 1))

    # L2-normalize copy for cosine spherical clustering
    norms = np.maximum(np.linalg.norm(z_orig, axis=1, keepdims=True), 1e-12)
    z_cos = z_orig / norms
    z_corr = z_orig.copy()

    rng = np.random.default_rng(seed)
    init_idx = rng.choice(n_cells, size=k_clust, replace=False)
    centroids = z_cos[init_idx].copy()
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

    prev_obj = np.inf
    r_mat = np.full((n_cells, k_clust), 1.0 / k_clust, dtype=np.float64)

    for _ in range(max_iter):
        z_cur_cos = z_corr / np.maximum(np.linalg.norm(z_corr, axis=1, keepdims=True), 1e-12)
        dist = 2.0 * (1.0 - np.clip(z_cur_cos @ centroids.T, -1.0, 1.0))

        # Diversity penalty: O_{bk} / E_{bk}
        obs_bk = phi.T @ r_mat + 1e-6
        cluster_mass = np.sum(r_mat, axis=0, keepdims=True) + 1e-6
        exp_bk = batch_priors[:, None] * cluster_mass
        div_ratio = np.power(np.divide(exp_bk, obs_bk), theta)
        cell_penalty = phi @ div_ratio

        logits = -dist / max(sigma, 1e-4) + np.log(np.maximum(cell_penalty, 1e-12))
        logits -= np.max(logits, axis=1, keepdims=True)
        exp_l = np.exp(logits)
        r_mat = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        # Update spherical centroids
        centroids = r_mat.T @ z_cur_cos
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

        # Mixture-of-experts batch correction per cluster
        z_next = z_orig.copy()
        design = np.hstack([np.ones((n_cells, 1), dtype=np.float64), phi])
        reg_eye = np.diag([0.0] + [ridge_lambda] * n_batches)

        for k_idx in range(k_clust):
            w_k = r_mat[:, k_idx]
            if np.sum(w_k) < 1e-6:
                continue
            wd = design * w_k[:, None]
            gram = wd.T @ design + reg_eye
            rhs = wd.T @ z_orig
            beta = np.linalg.solve(gram, rhs)
            batch_shift = phi @ beta[1:, :]
            z_next -= w_k[:, None] * batch_shift

        diff = float(np.mean(np.abs(z_next - z_corr)))
        z_corr = z_next
        obj = float(np.sum(r_mat * dist))
        if abs(prev_obj - obj) < tol and diff < tol:
            break
        prev_obj = obj

    return PCAResult(
        coordinates=z_corr,
        loadings=pca.loadings,
        singular_values=pca.singular_values,
        explained_variance=pca.explained_variance,
        explained_variance_ratio=pca.explained_variance_ratio,
        mean_vector=pca.mean_vector,
        batch_corrected=True,
    )


def summarize_batch_mixing(
    coords: NDArray[np.float64],
    batch_labels: Sequence[str],
    n_neighbors: int = 15,
) -> float:
    """Compute mean local batch-entropy mixing score in [0, 1] (1 = perfect mixing)."""
    n_cells = coords.shape[0]
    unique_batches = sorted(set(str(b) for b in batch_labels))
    n_batches = len(unique_batches)
    if n_batches <= 1 or n_cells < 4:
        return 1.0
    b_map = {b: i for i, b in enumerate(unique_batches)}
    b_arr = np.array([b_map[str(b)] for b in batch_labels], dtype=np.int64)
    k = max(2, min(n_neighbors, n_cells - 1))
    tree = cKDTree(coords)
    _, nn = tree.query(coords, k=k + 1)
    nn = nn[:, 1:]

    entropies = np.zeros(n_cells, dtype=np.float64)
    max_ent = np.log(float(n_batches))
    for i in range(n_cells):
        counts = np.bincount(b_arr[nn[i]], minlength=n_batches).astype(np.float64)
        probs = counts / np.sum(counts)
        nz = probs > 0.0
        entropies[i] = -float(np.sum(probs[nz] * np.log(probs[nz]))) / max_ent
    return float(np.mean(entropies))
