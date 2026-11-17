"""Tests for TranscriptomeDataset container, QC, Pearson residuals, HVGs, PCA, and Harmony."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from cyto_manifold.benchmark import generate_synthetic_tissue
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.preprocessing import (
    compute_pca,
    compute_pearson_residuals,
    compute_qc_metrics,
    filter_and_normalize,
    integrate_batches_harmony,
    select_highly_variable_genes,
    summarize_batch_mixing,
)
from cyto_manifold.types import TissueBenchmarkSpec


def test_dataset_sparse_roundtrip_and_checksum(tmp_path) -> None:
    rng = np.random.default_rng(1)
    x = sparse.csr_matrix(rng.poisson(2.0, size=(20, 12)).astype(np.float64))
    u = rng.poisson(1.0, size=(20, 12)).astype(np.float64)
    sp = rng.normal(size=(20, 2))
    ds = TranscriptomeDataset(X=x, spliced=x, unspliced=u, spatial=sp)
    assert ds.n_cells == 20
    assert ds.n_genes == 12
    assert ds.has_spatial()

    digest1 = ds.content_checksum()
    npz_path = ds.save_npz(tmp_path / "test_ds.npz")
    loaded = TranscriptomeDataset.load_npz(npz_path)
    assert loaded.content_checksum() == digest1
    assert np.allclose(loaded.X, ds.X)
    assert np.allclose(loaded.obsm["spatial"], sp)


def test_dataset_validation_rejects_negative_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TranscriptomeDataset(X=np.array([[-1.0, 2.0], [1.0, 1.0]]))

    with pytest.raises(ValueError, match="unique"):
        TranscriptomeDataset(
            X=np.ones((2, 2)),
            cell_ids=["C1", "C1"],
            gene_names=["G1", "G2"],
        )


def test_dataset_subset_preserves_layers_and_spatial() -> None:
    raw = generate_synthetic_tissue(TissueBenchmarkSpec(n_cells=40, n_genes=45, seed=3))
    c_mask = np.arange(raw.n_cells) % 2 == 0
    g_mask = np.arange(raw.n_genes) < 25
    sub = raw.subset(cell_mask=c_mask, gene_mask=g_mask)
    assert sub.n_cells == 20
    assert sub.n_genes == 25
    assert sub.obsm["spatial"].shape == (20, 2)
    assert sub.layers["spliced"].shape == (20, 25)


def test_qc_and_cp10k_library_size_invariant() -> None:
    raw = generate_synthetic_tissue(TissueBenchmarkSpec(n_cells=60, n_genes=45, seed=5))
    qc = compute_qc_metrics(raw, min_genes_per_cell=3, min_cells_per_gene=2, max_doublet_score=0.98)
    assert qc.n_cells_retained > 40
    assert qc.doublet_scores.shape == (60,)
    assert np.all((qc.doublet_scores >= 0.0) & (qc.doublet_scores <= 1.0))

    norm = filter_and_normalize(raw, qc=qc, target_sum=1e4)
    # Every retained cell has sum(expm1(X)) == 10,000 up to float precision
    reconstructed_sum = np.sum(np.expm1(norm.X), axis=1)
    assert np.allclose(reconstructed_sum, 1e4, rtol=1e-6)


def test_pearson_residuals_clipping_and_zero_mean_tendency() -> None:
    rng = np.random.default_rng(9)
    counts = rng.poisson(5.0, size=(50, 20)).astype(np.float64)
    res = compute_pearson_residuals(counts, theta=100.0, clip_sqrt_n=True)
    bound = np.sqrt(50.0)
    assert np.all(res <= bound + 1e-12)
    assert np.all(res >= -bound - 1e-12)
    assert abs(float(np.mean(res))) < 0.25


def test_hvg_pca_and_harmony_improves_batch_mixing() -> None:
    raw = generate_synthetic_tissue(
        TissueBenchmarkSpec(n_cells=140, n_genes=55, n_batches=2, batch_effect_strength=1.0, seed=12)
    )
    qc = compute_qc_metrics(raw, min_genes_per_cell=5, min_cells_per_gene=2, max_doublet_score=0.98)
    ds = filter_and_normalize(raw, qc=qc)
    hvg = select_highly_variable_genes(ds, n_top_genes=30)
    assert int(np.sum(hvg.highly_variable_mask)) == 30

    pca = compute_pca(ds, n_components=12, use_hvg=True)
    assert pca.coordinates.shape == (ds.n_cells, 12)
    assert np.all(pca.explained_variance_ratio >= 0.0)
    assert float(np.sum(pca.explained_variance_ratio)) <= 1.0 + 1e-6

    batches = ds.obs["batch"].tolist()
    mix_before = summarize_batch_mixing(pca.coordinates, batches)
    pca_corr = integrate_batches_harmony(pca, batches, n_clusters=4, seed=12)
    mix_after = summarize_batch_mixing(pca_corr.coordinates, batches)
    assert pca_corr.batch_corrected is True
    assert mix_after >= mix_before - 0.02
