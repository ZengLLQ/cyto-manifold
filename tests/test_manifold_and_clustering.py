"""Tests for fuzzy simplicial kNN manifold, MAGIC smoothing, and Leiden clustering."""

from __future__ import annotations

import numpy as np

from cyto_manifold.clustering import (
    adjusted_rand_index,
    cluster_dataset,
    compute_calinski_harabasz,
    compute_modularity,
    compute_silhouette_score,
    leiden_cluster,
    louvain_cluster,
    normalized_mutual_information,
    verify_clusters_connected,
)
from cyto_manifold.container import TranscriptomeDataset
from cyto_manifold.manifold import (
    build_knn_graph,
    compute_force_directed_layout,
    compute_spectral_embedding,
    compute_umap_layout,
    magic_diffusion_impute,
)


def _two_clique_graph(n_per_clique: int = 12) -> np.ndarray:
    """Create two dense cliques connected by a single weak bridge edge."""
    n = 2 * n_per_clique
    adj = np.zeros((n, n), dtype=np.float64)
    adj[:n_per_clique, :n_per_clique] = 1.0
    adj[n_per_clique:, n_per_clique:] = 1.0
    adj[0, n_per_clique] = 0.05
    adj[n_per_clique, 0] = 0.05
    np.fill_diagonal(adj, 0.0)
    return adj


def test_knn_fuzzy_simplicial_symmetry_and_range() -> None:
    rng = np.random.default_rng(7)
    pts = np.vstack([rng.normal(0.0, 0.4, (25, 6)), rng.normal(4.0, 0.4, (25, 6))])
    graph = build_knn_graph(pts, n_neighbors=10, metric="euclidean")
    w = graph.connectivities
    assert np.allclose(w, w.T, atol=1e-12)
    assert np.all((w >= 0.0) & (w <= 1.0))
    assert np.allclose(np.diag(w), 0.0)

    graph_cos = build_knn_graph(pts, n_neighbors=10, metric="cosine")
    assert np.allclose(graph_cos.connectivities, graph_cos.connectivities.T, atol=1e-12)


def test_spectral_umap_and_force_layouts() -> None:
    rng = np.random.default_rng(11)
    pts = np.vstack([rng.normal(-3.0, 0.4, (20, 5)), rng.normal(3.0, 0.4, (20, 5))])
    graph = build_knn_graph(pts, n_neighbors=8)
    spec = compute_spectral_embedding(graph.connectivities, n_components=2)
    assert spec.shape == (40, 2)
    assert np.all(np.isfinite(spec))

    umap_coords = compute_umap_layout(graph, n_epochs=60, seed=11)
    force_coords = compute_force_directed_layout(graph, n_iter=50, seed=11)
    assert umap_coords.shape == (40, 2)
    assert force_coords.shape == (40, 2)
    # Separated clusters should remain separated in 2D UMAP
    c0_center = np.mean(umap_coords[:20], axis=0)
    c1_center = np.mean(umap_coords[20:], axis=0)
    assert float(np.linalg.norm(c0_center - c1_center)) > 0.5


def test_magic_diffusion_reduces_dropout_variance() -> None:
    rng = np.random.default_rng(14)
    true_sig = np.vstack([np.full((25, 8), 5.0), np.full((25, 8), 1.0)])
    noisy = np.maximum(true_sig + rng.normal(0.0, 2.0, size=true_sig.shape), 0.0)
    ds = TranscriptomeDataset(X=noisy, spliced=noisy, unspliced=noisy * 0.4)
    graph = build_knn_graph(true_sig, n_neighbors=8)
    res = magic_diffusion_impute(ds, graph, t_steps=3)
    assert "Ms" in res and "Mu" in res
    var_before = float(np.var(noisy[:25, 0]))
    var_after = float(np.var(res["Ms"][:25, 0]))
    assert var_after < var_before


def test_leiden_and_louvain_recover_cliques_and_connectivity() -> None:
    adj = _two_clique_graph(12)
    labels_leiden = leiden_cluster(adj, resolution=1.0, seed=42)
    labels_louvain = louvain_cluster(adj, resolution=1.0, seed=42)
    truth = np.array([0] * 12 + [1] * 12, dtype=np.int64)

    assert adjusted_rand_index(truth, labels_leiden) == 1.0
    assert normalized_mutual_information(truth, labels_leiden) == 1.0
    assert adjusted_rand_index(truth, labels_louvain) == 1.0
    assert verify_clusters_connected(adj, labels_leiden)
    assert compute_modularity(adj, labels_leiden) > 0.4


def test_cluster_dataset_metrics() -> None:
    rng = np.random.default_rng(21)
    pts = np.vstack(
        [
            rng.normal(-4.0, 0.3, (18, 6)),
            rng.normal(0.0, 0.3, (18, 6)),
            rng.normal(4.0, 0.3, (18, 6)),
        ]
    )
    ds = TranscriptomeDataset(X=np.abs(pts) + 1.0)
    ds.set_obsm("X_pca", pts)
    graph = build_knn_graph(pts, n_neighbors=8)
    ds.set_obsp("connectivities", graph.connectivities)

    res = cluster_dataset(ds, resolution=0.9, method="leiden", seed=21)
    assert res.n_clusters == 3
    assert res.is_connected
    assert res.silhouette > 0.6
    assert compute_silhouette_score(pts, res.labels) > 0.6
    assert compute_calinski_harabasz(pts, res.labels) > 50.0
