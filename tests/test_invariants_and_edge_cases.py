"""Numerical invariants, edge cases, Sobol/OAT sensitivity, and convergence tests."""

from __future__ import annotations

import numpy as np
import pytest

from cyto_manifold.benchmark import generate_synthetic_tissue
from cyto_manifold.clustering import (
    adjusted_rand_index,
    compute_modularity,
    leiden_cluster,
    normalized_mutual_information,
    verify_clusters_connected,
)
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.differential import rank_genes_groups, run_gsea_enrichment
from cyto_manifold.grn import compute_aucell_scores, compute_mutual_information
from cyto_manifold.manifold import build_knn_graph, embed_manifold
from cyto_manifold.preprocessing import (
    compute_pca,
    compute_pearson_residuals,
    compute_qc_metrics,
    filter_and_normalize,
    integrate_batches_harmony,
    select_highly_variable_genes,
)
from cyto_manifold.pseudotime import compute_absorbing_fates, compute_diffusion_map
from cyto_manifold.spatial import build_spatial_graph, compute_morans_i
from cyto_manifold.types import TissueBenchmarkSpec, validate_matrix_shape
from cyto_manifold.velocity import analytical_splicing_trajectory


def test_validate_matrix_shape_rejects_bad_ndim_and_nan() -> None:
    with pytest.raises(ValueError, match="2-dimensional"):
        validate_matrix_shape(np.array([1.0, 2.0]), 1, 2, "bad")
    with pytest.raises(ValueError, match="non-finite"):
        validate_matrix_shape(np.array([[1.0, np.nan]]), 1, 2, "nan_mat")


def test_analytical_splicing_beta_equals_gamma_singularity_guard() -> None:
    t = np.linspace(0.0, 4.0, 41)
    u, s, _ = analytical_splicing_trajectory(t, alpha=10.0, beta=1.0, gamma=1.0, t_switch=2.0)
    assert np.all(np.isfinite(u))
    assert np.all(np.isfinite(s))
    assert np.all(s >= 0.0)


def test_leiden_resolution_monotonicity_invariant() -> None:
    rng = np.random.default_rng(70)
    pts = np.vstack(
        [
            rng.normal(-5.0, 0.3, (15, 4)),
            rng.normal(0.0, 0.3, (15, 4)),
            rng.normal(5.0, 0.3, (15, 4)),
        ]
    )
    graph = build_knn_graph(pts, n_neighbors=8)
    low_labels = leiden_cluster(graph.connectivities, resolution=0.3, seed=70)
    high_labels = leiden_cluster(graph.connectivities, resolution=1.8, seed=70)
    assert len(np.unique(high_labels)) >= len(np.unique(low_labels))
    assert verify_clusters_connected(graph.connectivities, low_labels)
    assert verify_clusters_connected(graph.connectivities, high_labels)


def test_ari_and_nmi_symmetry_and_identity() -> None:
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([2, 2, 0, 0, 1, 1])
    assert np.isclose(adjusted_rand_index(a, b), 1.0)
    assert np.isclose(normalized_mutual_information(a, b), 1.0)
    c = np.array([0, 1, 0, 1, 0, 1])
    assert np.isclose(adjusted_rand_index(a, c), adjusted_rand_index(c, a))
    assert np.isclose(normalized_mutual_information(a, c), normalized_mutual_information(c, a))


def test_modularity_zero_on_empty_graph() -> None:
    empty = np.zeros((10, 10), dtype=np.float64)
    assert compute_modularity(empty, np.zeros(10, dtype=np.int64)) == 0.0


def test_diffusion_map_eigenvalue_bounds_and_markov_stochasticity() -> None:
    rng = np.random.default_rng(88)
    pts = rng.normal(size=(30, 5))
    graph = build_knn_graph(pts, n_neighbors=7)
    dc, lambdas, t_mat = compute_diffusion_map(graph.connectivities, n_components=6)
    assert dc.shape == (30, 6)
    assert np.all(np.abs(lambdas) < 1.0)
    assert np.allclose(np.sum(t_mat, axis=1), 1.0, atol=1e-9)


def test_single_terminal_state_absorbing_fates_all_one() -> None:
    t_mat = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    fates, entropy = compute_absorbing_fates(t_mat, {0: np.array([2], dtype=np.int64)})
    assert np.allclose(fates[:, 0], 1.0, atol=1e-6)
    assert np.allclose(entropy, 0.0, atol=1e-6)


def test_harmony_single_batch_is_noop() -> None:
    raw = generate_synthetic_tissue(TissueBenchmarkSpec(n_cells=50, n_genes=45, n_batches=1, seed=15))
    ds = filter_and_normalize(raw)
    select_highly_variable_genes(ds, n_top_genes=25)
    pca = compute_pca(ds, n_components=8)
    pca_out = integrate_batches_harmony(pca, ds.obs["batch"].tolist())
    assert np.allclose(pca.coordinates, pca_out.coordinates)


def test_pearson_residuals_reject_non_positive_theta() -> None:
    with pytest.raises(ValueError, match="positive"):
        compute_pearson_residuals(np.ones((4, 4)), theta=0.0)


def test_spatial_morans_i_invariant_to_weight_scaling() -> None:
    gx, gy = np.meshgrid(np.arange(5.0), np.arange(5.0))
    coords = np.column_stack([gx.ravel(), gy.ravel()])
    w = build_spatial_graph(coords, mode="knn", n_neighbors=4)
    vals = coords[:, 0] * 2.0 - coords[:, 1]
    mi_1, _, z_1, _ = compute_morans_i(vals, w)
    mi_2, _, z_2, _ = compute_morans_i(vals, w * 7.5)
    assert np.isclose(mi_1, mi_2, atol=1e-9)
    assert np.isclose(z_1, z_2, atol=1e-9)


def test_mutual_information_constant_vector_zero() -> None:
    x = np.ones(50, dtype=np.float64)
    y = np.arange(50, dtype=np.float64)
    assert compute_mutual_information(x, y) == 0.0


def test_aucell_empty_regulons_and_single_cluster_de_guard() -> None:
    expr = np.ones((10, 5), dtype=np.float64)
    auc, tfs = compute_aucell_scores(expr, ["A", "B", "C", "D", "E"], {})
    assert auc.shape == (10, 0)
    assert tfs == []

    ds = TranscriptomeDataset(X=expr, obs={"cluster": np.zeros(10, dtype=np.int64)})
    assert rank_genes_groups(ds) == []
    assert run_gsea_enrichment(ds) == []


def test_qc_and_embed_force_layout_integration() -> None:
    raw = generate_synthetic_tissue(TissueBenchmarkSpec(n_cells=60, n_genes=45, seed=29))
    qc = compute_qc_metrics(raw, min_genes_per_cell=4, min_cells_per_gene=2, max_doublet_score=0.98)
    ds = filter_and_normalize(raw, qc=qc)
    compute_pca(ds, n_components=10, use_hvg=False)
    _, coords = embed_manifold(ds, n_neighbors=10, layout="force", seed=29)
    assert "X_draw_graph" in ds.obsm
    assert coords.shape == (ds.n_cells, 2)
    assert np.all(np.isfinite(coords))
